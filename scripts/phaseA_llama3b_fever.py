#!/usr/bin/env python3
# phaseA_llama3b_fever.py  —— 微调 Llama-3-3B-Instruct 在 FEVER 二分类

import os, argparse
from typing import Dict
from datasets import load_dataset, Dataset
from huggingface_hub import login as hf_login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ------------------------- Wiki 缓存 -------------------------
class WikiCache:
    def __init__(self, cache_dir=None):
        wiki = load_dataset("fever", "wiki_pages", cache_dir=cache_dir)["wikipedia_pages"]
        self._idx = {row["id"]: i for i, row in enumerate(wiki)}
        self._data, self._cache = wiki, {}

    def sent(self, page_id, sent_id: int) -> str:
        key = str(page_id)
        if key not in self._cache:
            if key not in self._idx:
                return ""
            rec = self._data[self._idx[key]]
            self._cache[key] = {
                int(line.split("\t", 1)[0]): line.split("\t", 1)[1]
                for line in rec["lines"].split("\n") if line
            }
        return self._cache[key].get(sent_id, "")

# ------------------------- Prompt 构造 -------------------------
def build_prompt(sys_msg, evidence, claim):
    return (
        f"<s>[INST] {sys_msg}\n"
        f"Evidence: {evidence}\n"
        f"Claim: {claim}\n"
        "Question: Is the claim supported or refuted by the evidence?\n"
        "Answer:[/INST] "
    )

def prepare_dataset(keep_nei=False, cache_dir=None) -> Dataset:
    raw  = load_dataset("fever", "v1.0", split="train", cache_dir=cache_dir)
    wiki = WikiCache(cache_dir)
    sys_msg = (
        "<<SYS>>\n"
        "You are a fact-checking assistant.\n"
        "Given EVIDENCE and a CLAIM, reply with exactly one token: "
        + ("SUPPORTED, REFUTED, or NOT ENOUGH INFO.\n" if keep_nei
           else "SUPPORTED or REFUTED.\n")
        + "Do not output anything else.\n"
        "<</SYS>>"
    )

    def convert(ex):
        lab = ex["label"]
        if lab == "NOT ENOUGH INFO" and not keep_nei:
            return None
        if lab == "NOT ENOUGH INFO":
            ev, y = "No evidence provided.", "NOT ENOUGH INFO"
        else:
            ev = wiki.sent(ex["evidence_id"], ex["evidence_sentence_id"])
            y  = "SUPPORTED" if lab == "SUPPORTS" else "REFUTED"
        return {"text": build_prompt(sys_msg, ev, ex["claim"]) + y + " </s>"}

    ds = raw.map(convert, remove_columns=raw.column_names)
    return ds.filter(lambda x: x is not None)

# ------------------------- Trainer --------------------------
def build_trainer(cfg) -> SFTTrainer:
    if cfg.token or os.getenv("HF_TOKEN"):
        hf_login(cfg.token or os.getenv("HF_TOKEN"))

    tok = AutoTokenizer.from_pretrained(cfg.model_id, add_eos_token=False, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, device_map="auto", torch_dtype="auto")
    model = get_peft_model(
        base,
        LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                   bias="none", lora_dropout=0.05)
    )

    train_ds = prepare_dataset(cfg.keep_nei, cfg.cache_dir)
    targs = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch,      # 默认 16
        gradient_accumulation_steps=cfg.grad_acc,   # 默认 2  → 有效 32
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=cfg.fp16, bf16=cfg.bf16,
        logging_steps=100, save_strategy="epoch",
        gradient_checkpointing=cfg.grad_ckpt,
    )

    return SFTTrainer(
        model=model, tokenizer=tok, train_dataset=train_ds,
        max_seq_length=512, dataset_text_field="text", args=targs
    )

# ------------------------- CLI ----------------------------
def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--token", default=None, help="HF token or HF_TOKEN env")
    ap.add_argument("--output_dir", default="./phaseA_llama3B")
    ap.add_argument("--batch", type=int, default=16)    # 单卡 H100 建议 16
    ap.add_argument("--grad_acc", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--keep_nei", action="store_true")
    ap.add_argument("--cache_dir", default=None)
    return ap.parse_args()

# ------------------------- MAIN ---------------------------
if __name__ == "__main__":
    cfg = cli()
    trainer = build_trainer(cfg)
    trainer.train()
    trainer.save_model(cfg.output_dir)
