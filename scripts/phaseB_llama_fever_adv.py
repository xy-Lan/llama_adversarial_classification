#!/usr/bin/env python3
# Phase B: CE + α·KL 对抗一致性微调

import os, argparse, pandas as pd, torch, torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer)
from peft import PeftModel, LoraConfig, get_peft_model
from huggingface_hub import login as hf_login

# ---------- 1. 解析 CSV ----------
def load_adv_pairs(path: str, keep_changed: bool = False) -> Dataset:
    df = pd.read_csv(path)
    if not keep_changed:
        df = df[df["agreed_labels"] == 0]          # 只保留语义一致
    df = df.reset_index(drop=True)

    def explode(row):
        return {
            "text":      [row.original_samples, row.adversarial_samples],
            "pair_id":   [row.name, row.name],     # 行号作 id
            "is_adv":    [0, 1],
            "semantic":  [row.agreed_labels]*2,    # 0 or 1/2
            "labels":    [-100, -100],             # 无监督
        }

    return Dataset.from_pandas(df).map(explode, batched=False,
                                       remove_columns=list(df.columns))

# ---------- 2. 加载 Phase-A 权重 ----------
def load_lora_model(base_id: str, phaseA_dir: str, bf16=False, fp16=False):
    dtype = "bfloat16" if bf16 else ("float16" if fp16 else "float32")
    base  = AutoModelForCausalLM.from_pretrained(
                base_id, device_map="auto", torch_dtype=dtype)
    return PeftModel.from_pretrained(base, phaseA_dir)

# ---------- 3. 自定义 Trainer ----------
class AdvTrainer(Trainer):
    def __init__(self, alpha=1.0, **kw):
        super().__init__(**kw)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        pair_id  = inputs.pop("pair_id")
        semantic = inputs.pop("semantic")          # 0 preserve / 1/2 change
        logits   = model(**inputs).logits[:, -1]   # [B, vocab]

        # ① 交叉熵 — 只有 FEVER label 样本才有 labels ≠ -100
        labels   = inputs["labels"]
        mask_lbl = labels != -100
        loss_ce  = (F.cross_entropy(
                        logits[mask_lbl], labels[mask_lbl]) if mask_lbl.any()
                    else torch.tensor(0., device=logits.device))

        # ② KL 一致性 — 同 pair 且 semantic == 0
        idx_o = torch.arange(0, logits.size(0), 2, device=logits.device)
        idx_a = idx_o + 1
        same  = (semantic[idx_o] == 0)
        if same.any():
            p = F.log_softmax(logits[idx_o][same], -1)
            q = F.softmax     (logits[idx_a][same], -1)
            loss_k = F.kl_div(p, q, reduction="batchmean")
        else:
            loss_k = torch.tensor(0., device=logits.device)

        loss = loss_ce + self.alpha * loss_k
        return (loss, logits) if return_outputs else loss

# ---------- 4. CLI ----------
def cli():
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
    return ap.parse_args()

# ---------- MAIN ----------
if __name__ == "__main__":
    cfg = cli()
    if cfg.token or os.getenv("HF_TOKEN"):
        hf_login(cfg.token or os.getenv("HF_TOKEN"))

    tok = AutoTokenizer.from_pretrained(cfg.base_model, add_eos_token=False,
                                        use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # 1) FEVER 监督部分
    fever_train = load_dataset("fever", "v1.0", split="train")
    fever_train = fever_train.filter(lambda x: x["label"] != "NOT ENOUGH INFO")\
                             .map(lambda x: {"labels": tok.convert_tokens_to_ids(
                                  "SUPPORTED" if x["label"]=="SUPPORTS" else "REFUTED")})
    fever_train = fever_train.remove_columns(
                       [c for c in fever_train.column_names if c != "text" and c!="labels"])
    # 2) 对抗对
    pairs_ds = load_adv_pairs(cfg.pairs_csv)
    # 3) 混合
    train_ds = concatenate_datasets([fever_train, pairs_ds])

    model = load_lora_model(cfg.base_model, cfg.phaseA_dir,
                            bf16=cfg.bf16, fp16=cfg.fp16)

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
            save_strategy="epoch",
            logging_steps=50,
        ),
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
