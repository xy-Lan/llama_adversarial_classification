"""
phaseA_llama1b_fever.py  ──────────────────────────────────────────────

End‑to‑end **Phase A** fine‑tuning script, now defaulting to
`meta‑llama/Llama‑3.2‑1B‑Instruct` and supporting a Hugging Face token for
private‑repo download.

────────────────────────────────────────────────────────────────────────────
Quick start
────────────────────────────────────────────────────────────────────────────
# 1)  Prepare env (CUDA 11.8 example)  ↘
conda create -n fever_llama3 python=3.10 -y && conda activate fever_llama3
pip install torch==2.2.2+cu118 -f https://download.pytorch.org/whl/cu118
pip install transformers==4.40.1 peft==0.10.0 datasets==2.18.0 trl==0.8.6 huggingface_hub==0.22.2

# 2)  Run   (⚠️ **Never** hard‑code your token in notebooks)
export HF_TOKEN="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"
python phaseA_llama1b_fever.py \
       --token "$HF_TOKEN" \
       --model_id "meta-llama/Llama-3.2-1B-Instruct" \
       --output_dir ./llama3_phaseA \
       --batch 16 --grad_acc 2 --epochs 1 --lr 2e-4
"""
from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from huggingface_hub import login as hf_login
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("phaseA")

# ──────────────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────────────
SYS = (
    "<<SYS>>\n"
    "You are a fact‑checking assistant.\n"
    "Given EVIDENCE and a CLAIM, reply with exactly one token: SUPPORTED or REFUTED.\n"
    "Do not output anything else.\n"
    "<</SYS>>\n\n"
)
USER_Q = "Question: Is the claim supported or refuted by the evidence?\nAnswer:"

# ──────────────────────────────────────────────────────────────────────────
# Wikipedia sentence cache (lazy)
# ──────────────────────────────────────────────────────────────────────────
class WikiCache:
    def __init__(self):
        self._data = load_dataset("fever", "wiki_pages", split="train")
        self._title2row = {row["title"]: i for i, row in enumerate(self._data)}
        self._cache: Dict[str, Dict[int, str]] = {}

    def sent(self, title: str, sid: int) -> str:
        if title not in self._cache:
            rec = self._data[self._title2row[title]]
            sent_map = {
                int(line.split("\t", 1)[0]): line.split("\t", 1)[1]
                for line in rec["lines"].split("\n") if line
            }
            self._cache[title] = sent_map
        return self._cache[title][sid]

WIKI = WikiCache()

# ──────────────────────────────────────────────────────────────────────────
# Helper to build evidence‑aware prompt
# ──────────────────────────────────────────────────────────────────────────

def build_prompt(claim: str, ev_groups: List[List[int]]) -> str:
    group = ev_groups[0]  # first evidence set
    sents = [WIKI.sent(t[2], t[3]) for t in group][:3]
    evidence_text = " ".join(sents)
    return (
        f"Evidence: {evidence_text}\n"
        f"Claim: {claim}\n"
        f"{USER_Q}"
    )

# ──────────────────────────────────────────────────────────────────────────
# Convert FEVER train → prompt jsonl (binary, NEI removed)
# ──────────────────────────────────────────────────────────────────────────

def prepare_jsonl(path: Path):
    if path.exists():
        log.info("[cache] %s exists", path)
        return
    ds = load_dataset("fever", split="train")
    with path.open("w", encoding="utf-8") as fw:
        for ex in ds:
            if ex["label"] == "NOT ENOUGH INFO":
                continue
            label = "SUPPORTED" if ex["label"] == "SUPPORTS" else "REFUTED"
            prompt = build_prompt(ex["claim"], ex["evidence"])
            inst = f"<s>[INST] {SYS}{prompt} [/INST] {label} </s>"
            fw.write(json.dumps({"text": inst}) + "\n")
    log.info("FEVER binary jsonl saved → %s", path)

# ──────────────────────────────────────────────────────────────────────────
# Build trainer
# ──────────────────────────────────────────────────────────────────────────

def build_trainer(cfg) -> SFTTrainer:
    if cfg.token:
        hf_login(token=cfg.token)
    tok = AutoTokenizer.from_pretrained(cfg.model_id, add_eos_token=False, use_fast=False)
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, device_map="auto")

    lora = get_peft_model(base, LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"]))

    train_ds = load_dataset("json", data_files=str(cfg.data_jsonl), split="train")

    targs = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch,
        gradient_accumulation_steps=cfg.grad_acc,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=100,
        save_strategy="epoch",
        gradient_checkpointing=cfg.grad_ckpt,
    )

    return SFTTrainer(model=lora, tokenizer=tok, train_dataset=train_ds, args=targs, max_seq_length=512)

# ──────────────────────────────────────────────────────────────────────────
# Evaluate on FEVER dev (binary)
# ──────────────────────────────────────────────────────────────────────────

def evaluate(trainer: SFTTrainer) -> float:
    dev = load_dataset("fever", split="validation").filter(lambda e: e["label"] != "NOT ENOUGH INFO")
    correct = 0
    tok = trainer.tokenizer
    model = trainer.model
    for ex in dev:
        prompt = build_prompt(ex["claim"], ex["evidence"])
        full = f"<s>[INST] {SYS}{prompt} [/INST]"
        inp = tok(full, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=1)
        pred = tok.decode(out[0], skip_special_tokens=True).split()[-1].upper()
        gold = "SUPPORTED" if ex["label"] == "SUPPORTS" else "REFUTED"
        correct += int(pred == gold)
    return correct / len(dev)

# ──────────────────────────────────────────────────────────────────────────
# CLI glue
# ──────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HuggingFace access token")
    p.add_argument("--model_id", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--output_dir", default="llama3_phaseA")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--grad_acc", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad_ckpt", action="store_true")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    cfg = parse_args()
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    cfg.data_jsonl = Path(cfg.output_dir) / "fever_binary_train.jsonl"
    prepare_jsonl(cfg.data_jsonl)

    trainer = build_trainer(cfg)
    trainer.train()
    trainer.save_model(cfg.output_dir)

    acc = evaluate(trainer)
    log.info("Clean FEVER‑dev accuracy: %.4f", acc)
    with open(Path(cfg.output_dir) / "metrics.json", "w") as f:
        json.dump({"clean_dev_accuracy": acc}, f, indent=2)

if __name__ == "__main__":
    main()
