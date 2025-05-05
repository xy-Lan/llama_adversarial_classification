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
    """读取 pairs CSV 并展开为 2×行：
       - original_samples → is_adv=0
       - adversarial_samples → is_adv=1
    """
    df = pd.read_csv(csv_path)
    if not keep_changed:
        df = df[df["agreed_labels"] == 0]

    df_orig = df[["original_samples", "agreed_labels"]].copy()
    df_orig.columns = ["text", "semantic"]           # semantic: 0=相同，1=不同
    df_orig["is_adv"] = 0

    df_adv = df[["adversarial_samples", "agreed_labels"]].copy()
    df_adv.columns = ["text", "semantic"]
    df_adv["is_adv"] = 1

    big = pd.concat([df_orig, df_adv], ignore_index=True)
    big["pair_id"] = big.index // 2                  # 同一对共享 id
    big["labels"]  = -100                            # Phase B 不算 CE
    return Dataset.from_pandas(big, preserve_index=False)

# ========== 2. Data collator ==========
def adv_collator(features):
    input_ids = [torch.tensor(f["input_ids"]) for f in features]
    pad_id = tok.pad_token_id
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    attention_mask = (input_ids != pad_id).long()

    def stack(name):                                   # 不再担心缺列
        return torch.tensor([f[name] for f in features])

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
        pair_id  = inputs.pop("pair_id")
        semantic = inputs.pop("semantic")
        logits   = model(**inputs).logits[:, -1]       # 只取最后一个 token

        # 1) Cross‑entropy（此处 labels 全为 -100，可跳过）
        labels   = inputs["labels"]
        mask_lbl = labels != -100
        loss_ce  = F.cross_entropy(logits[mask_lbl], labels[mask_lbl]) if mask_lbl.any() else 0.

        # 2) KL：同一 pair、semantic==0 时计算
        idx_o = torch.arange(0, logits.size(0), 2, device=logits.device)
        idx_a = idx_o + 1
        same  = semantic[idx_o] == 0
        loss_kl = 0.
        if same.any():
            p = F.log_softmax(logits[idx_o][same], -1)
            q = F.softmax     (logits[idx_a][same], -1)
            loss_kl = F.kl_div(p, q, reduction="batchmean")

        loss = loss_ce + self.alpha * loss_kl
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
                tokens[key] = examples[key]
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

    # Sanity‑check
    assert all("pair_id" in row for row in train_ds)

    # 5‑4. Model (+LoRA)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, device_map="auto", torch_dtype="auto")
    model = PeftModel.from_pretrained(base_model, cfg.phaseA_dir) \
            if cfg.phaseA_dir else base_model

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
            logging_steps=20,
            save_strategy="epoch",
            dataloader_drop_last=True,      # batch 必须偶数
        )
    )

    # 5‑6. Train + save
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"✅ Phase B 权重已保存到：{cfg.output_dir}")
