#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase B 训练（只用 700 对 adversarial 样本）
------------------------------------------------
• 输入：pairs_csv        —— 需含列 original_samples, adversarial_samples, agreed_labels
• 目标：让模型对“语义未变、标签翻转”的文本对保持一致预测（KL loss）
"""
import os, argparse, pandas as pd, torch, torch.nn.functional as F
from datasets import Dataset, Value
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments)
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence

# ========== 1. CSV → Dataset ==========
def load_adv_pairs(csv_path: str, keep_changed=False) -> Dataset:
    """读取预处理后的扁平 CSV，每条样本包含：
       - text
       - is_adv
       - semantic
       - pair_id
    可选地过滤掉 semantic != 0 的样本
    """
    df = pd.read_csv(csv_path)

    if not keep_changed:
        df = df[df["semantic"] == 0]  # 只保留语义未变化的样本

    df["labels"] = -100  # Phase B 不训练分类器，只训练区分对抗样本
    return Dataset.from_pandas(df, preserve_index=False)


# ========== 2. Data collator ==========
def adv_collator(features):
    input_ids = [f["input_ids"].clone() for f in features]
    pad_id = tok.pad_token_id
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    attention_mask = (input_ids != pad_id).long()

    # ✅ 在这里定义改进的 stack 函数
    def stack(name):
        for i, f in enumerate(features):
            if name not in f:
                print(f"[DEBUG] Sample {i} is missing field '{name}'. Available keys: {list(f.keys())}")
        values = [f[name] for f in features if name in f]
        if len(values) < len(features):
            raise ValueError(f"Some samples in the batch are missing the '{name}' field.")
        return torch.stack(values) if isinstance(values[0], torch.Tensor) else torch.tensor(values)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": stack("labels"),
        "pair_id": stack("pair_id"),
        "semantic": stack("semantic"),
        "is_adv": stack("is_adv"),
    }

# ========== 3. 自定义 Trainer ==========
class AdvTrainer(Trainer):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        print(">> DEBUG: entered compute_loss 进入！！！！")
        print(">> DEBUG: inside compute_loss; id=", id(self))

        # ----- 取出自定义字段 -----
        pair_id = inputs.pop("pair_id")
        semantic = inputs.pop("semantic")

        # ----- 前向 -----
        logits = model(**inputs).logits[:, -1]  # [B, vocab]

        # 1) CE：挂在 logits 上的 0
        loss_ce = logits.sum() * 0.0

        # 2) KL
        idx_o = torch.arange(0, logits.size(0), 2, device=logits.device)
        idx_a = idx_o + 1
        same = semantic[idx_o] == 0
        if same.any():
            p = F.log_softmax(logits[idx_o][same], -1)
            q = F.softmax(logits[idx_a][same], -1)
            loss_kl = F.kl_div(p, q, reduction="batchmean")
        else:
            loss_kl = logits.sum() * 0.0  # 仍挂在 logits 上

        dummy = logits.sum() * 0.0  # 再保险 1 层

        loss = dummy + loss_ce + self.alpha * loss_kl

        # ========= 这里插入调试 =========
        # print("   logits.requires_grad =", logits.requires_grad)
        # print("   loss.requires_grad   =", loss.requires_grad)
        # =================================

        return (loss, logits) if return_outputs else loss


# ========== 4. CLI ==========
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True,default="meta-llama/Llama-3.2-1B-Instruct",
                    help="基座模型或 HuggingFace 路径")
    ap.add_argument("--phaseA_dir",  default="./phaseA_llama3",
                    help="Phase A LoRA 权重目录（留空=不用）")
    ap.add_argument("--pairs_csv", required=True,default="./data/train.csv",
                    help="700 对 adversarial 样本 csv")
    ap.add_argument("--output_dir", required=True,default="./phaseB_llama3_fromA",
                    help="保存 Phase B 权重的目录")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="KL loss 系数")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad_acc", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--token", default=None,
                    help="HuggingFace token（如私有模型需登录）")
    return ap.parse_args()

# ========== 5. Main ==========
if __name__ == "__main__":
    cfg = get_args()

    # 5‑1. tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # 5‑2. Dataset：只用 pairs
    train_ds = load_adv_pairs(cfg.pairs_csv)
    # ★ cast dtype（int64）
    for col in ("semantic", "is_adv", "pair_id"):
        train_ds = train_ds.cast_column(col, Value("int64"))

    # 5‑3. Tokenisation
    def tok_fn(examples):
        tokens = tok(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        tokens["labels"] = tokens["input_ids"].copy()

        # 保留其他字段
        for key in ["pair_id", "semantic", "is_adv"]:
            if key in examples:
                tokens[key] = list(examples[key])
        return tokens


    train_ds = train_ds.map(
        tok_fn,
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=False
    )
    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels",
                 "pair_id", "semantic", "is_adv"]
    )
    # print("Example keys after tokenization:", train_ds[0].keys())

    required_fields = ["input_ids", "attention_mask", "labels", "pair_id", "semantic", "is_adv"]
    train_ds = train_ds.filter(lambda ex: all(k in ex and ex[k] is not None for k in required_fields))

    # Sanity‑check
    assert all("pair_id" in row for row in train_ds)

    # ========== 5‑4. Model (+LoRA)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, device_map="auto", torch_dtype="auto")

    model = PeftModel.from_pretrained(base_model, cfg.phaseA_dir) \
        if cfg.phaseA_dir else base_model

    model.train()

    # ✅ 解冻全部参数（必要时 LoRA adapter 默认已解冻）
    for param in model.parameters():
        param.requires_grad = True

    # ✅ 打印确认
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"✅ Trainable parameters: {len(trainable)}")

    # 5‑5. Trainer
    trainer = AdvTrainer(
        alpha=cfg.alpha,
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        data_collator=adv_collator,
        args=TrainingArguments(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.batch,
            gradient_accumulation_steps=cfg.grad_acc,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.lr,
            warmup_ratio=0.05,
            fp16=cfg.fp16,
            bf16=cfg.bf16,
            logging_dir=f"{cfg.output_dir}/logs",  # ← 日志路径
            logging_strategy="steps",  # ← 每多少步记录一次
            logging_steps=10,  # ← 每 10 步记录一次
            save_strategy="epoch",  # ← 每个 epoch 保存一次
            save_total_limit=1,  # ← 只保留最近一次（可选）
            dataloader_drop_last=True,
            remove_unused_columns=False  # ← 保留自定义字段
        )
    )

    # 5‑6. Train + save
    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    print(f"✅ 仅保存 LoRA adapter 和 tokenizer 到：{cfg.output_dir}")
    save_strategy = "epoch"
    save_total_limit = 1
