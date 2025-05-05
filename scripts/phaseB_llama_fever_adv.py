#!/usr/bin/env python3
# phaseB_llama_fever_adv.py  (2025-04-25)
# -----------------------------------------------------------
# Phase-B: LoRA 继续微调 —— 交叉熵(CE) + α·KL 对抗一致性
# 数据：FEVER v1.0 train（去 NEI）＋ 700 对 orig/adv
# -----------------------------------------------------------
import os, argparse, pandas as pd, torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset, concatenate_datasets, Value
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer)
from peft import PeftModel
from huggingface_hub import login as hf_login

# ---------- Part 1. 复用 Phase-A 的 WikiCache 与 prompt ----------
from phaseA_llama1b_fever import WikiCache         # 保证在同目录
# 若你把 WikiCache 抽到 util.py，请从 util 导入

SYS_MSG = (
    "<<SYS>>\nYou are a fact-checking assistant.\n"
    "Given EVIDENCE and a CLAIM, reply with exactly one token: "
    "SUPPORTED or REFUTED.\nDo not output anything else.\n<</SYS>>"
)

def build_prompt(evidence: str, claim: str) -> str:
    return (
        f"<s>[INST] {SYS_MSG}\n"
        f"Evidence: {evidence}\n"
        f"Claim: {claim}\n"
        "Question: Is the claim supported or refuted by the evidence?\n"
        "Answer:[/INST] "
    )

# ---------- Part 2. FEVER → 生成 text 列 + dummy labels ----------
def get_fever_dataset(cache_dir=None):
    wiki = WikiCache(cache_dir)
    raw  = load_dataset("fever", "v1.0", split="train", cache_dir=cache_dir, trust_remote_code=True)
    raw  = raw.filter(lambda x: x["label"] != "NOT ENOUGH INFO")

    def to_prompt(ex):
        ev = wiki.sent(ex["evidence_id"], ex["evidence_sentence_id"])
        lab_tok = "SUPPORTED" if ex["label"] == "SUPPORTS" else "REFUTED"
        return {
            "text": build_prompt(ev, ex["claim"]) + lab_tok + " </s>",
            "labels": -100,         # Phase-B 不对 FEVER 行算 CE
            "pair_id": -1,  # 单独样本，用 -1 占位
            "semantic": 1,  # 1 表示“不要求对齐”，compute_loss 时 same==0 才算 KL
            "is_adv": 0,  # 0=orig，1=adv，可随意
        }

    return raw.map(
        to_prompt,
        remove_columns=raw.column_names,          # 只留下 text + labels
    )

# ---------- Part 3. 700 对 adversarial CSV ----------
def load_adv_pairs(csv_path: str, keep_changed=False) -> Dataset:
    df = pd.read_csv(csv_path)
    if not keep_changed:
        df = df[df["agreed_labels"] == 0]

    # -------- 用 pandas 直接展开成两倍行数 --------
    df_orig = df[["original_samples", "agreed_labels"]].copy()
    df_orig.columns = ["text", "semantic"]
    df_orig["is_adv"] = 0

    df_adv  = df[["adversarial_samples", "agreed_labels"]].copy()
    df_adv.columns = ["text", "semantic"]
    df_adv["is_adv"] = 1

    big = pd.concat([df_orig, df_adv], ignore_index=True)
    big["pair_id"] = big.index // 2         # 相同对同一个 id
    big["labels"]  = -100

    # 直接转 Dataset（全部列长度一致）
    return Dataset.from_pandas(big, preserve_index=False)

# ---------- Part 4. 自定义 Trainer with CE + α·KL ----------
class AdvTrainer(Trainer):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        pair_id  = inputs.pop("pair_id")
        semantic = inputs.pop("semantic")
        logits   = model(**inputs).logits[:, -1]        # 仅最后 token

        # 1) 主任务 CE — 我们只在 orig 有标签时才算，这里 labels 全为 -100
        labels   = inputs["labels"]
        mask_lbl = labels != -100
        loss_ce  = F.cross_entropy(logits[mask_lbl], labels[mask_lbl]) if mask_lbl.any() else 0.

        # 2) KL 一致性 — 语义保留且成对
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

# ---------- Part 5. CLI & Main ----------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--phaseA_dir", default="./phaseA_llama3")
    ap.add_argument("--pairs_csv", default="./data/train.csv")
    ap.add_argument("--output_dir", default="./phaseB_llama3")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad_acc", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--token", default=None)
    ap.add_argument("--cache_dir", default=None)
    return ap.parse_args()

if __name__ == "__main__":
    cfg = get_args()
    if cfg.token or os.getenv("HF_TOKEN"):
        hf_login(cfg.token or os.getenv("HF_TOKEN"))

    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # -------- datasets --------
    pairs_ds  = load_adv_pairs(cfg.pairs_csv)
    pairs_ds = pairs_ds.cast_column("semantic", Value("int64"))  # float → int
    fever_ds = get_fever_dataset(cfg.cache_dir)
    # 只添加不存在的列，避免重复
    if "is_adv" not in fever_ds.column_names:
        fever_ds = fever_ds.add_column("is_adv", [0] * len(fever_ds))
    if "pair_id" not in fever_ds.column_names:
        fever_ds = fever_ds.add_column("pair_id", list(range(len(fever_ds))))

    # 可选：显式强制 dtypes
    fever_ds = fever_ds.cast_column("semantic", Value("int64"))
    fever_ds = fever_ds.cast_column("is_adv", Value("int64"))
    fever_ds = fever_ds.cast_column("pair_id", Value("int64"))

    train_ds  = concatenate_datasets([fever_ds, pairs_ds])


    # ========== ① 在这里：先 tokenization，然后打印一个样本 ==========
    def tok_fn(batch):
        out = tok(batch["text"],
                  truncation=True,
                  padding=False)  # 让 DataCollatorWithPadding 再统一 pad
        out["labels"] = batch["labels"]  # 把原 label 搬过来
        # 其余自定义字段原样保留
        for fld in ("pair_id", "semantic", "is_adv"):
            if fld in batch:
                out[fld] = batch[fld]
        return out

    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])

    # —— 快速调试：看第一条样本结构 ——
    from itertools import islice

    print("⟹ 调试样本:", next(islice(train_ds, 0, 1)))
    print(train_ds[0]["pair_id"], train_ds[1]["pair_id"],
          train_ds[0]["is_adv"], train_ds[1]["is_adv"])

    # -------- model with Phase-A LoRA --------
    base  = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, device_map="auto", torch_dtype="auto")
    model = PeftModel.from_pretrained(base, cfg.phaseA_dir)

    trainer = AdvTrainer(
        alpha=cfg.alpha,
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        args=TrainingArguments(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.batch,
            gradient_accumulation_steps=cfg.grad_acc,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.lr,
            warmup_ratio=0.05,
            fp16=cfg.fp16,
            bf16=cfg.bf16,
            logging_steps=50,
            save_strategy="epoch",
            dataloader_shuffle=False,  # 关键：保持顺序
            drop_last=True  # batch 一定是偶数
        )
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)

