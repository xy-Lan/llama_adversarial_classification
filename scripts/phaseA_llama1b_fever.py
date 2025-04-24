#!/usr/bin/env python3
# phaseA_llama_fever.py
# -----------------------------------------------------------
# 微调 Llama-3-1B-Instruct 在 FEVER (claim+evidence) 二分类任务
# -----------------------------------------------------------
import os, argparse, json, tempfile
from typing import Dict, List

from datasets import load_dataset, Dataset
from huggingface_hub import login as hf_login

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model
from trl  import SFTTrainer


# ------------------------- 数据工具 -------------------------
class WikiCache:
    def __init__(self, cache_dir=None):
        wiki = load_dataset("fever", "wiki_pages", cache_dir=cache_dir)["wikipedia_pages"]
        self._idx = {row["id"]: i for i, row in enumerate(wiki)}   # 用字符串键
        self._data = wiki
        self._cache = {}

    def sent(self, page_id, sent_id: int) -> str:
        page_key = str(page_id)                                    # ←★
        if page_key not in self._cache:
            if page_key not in self._idx:
                return ""  # 页面缺失
            rec = self._data[self._idx[page_key]]
            self._cache[page_key] = {
                int(line.split("\t", 1)[0]): line.split("\t", 1)[1]
                for line in rec["lines"].split("\n") if line
            }
        return self._cache[page_key].get(sent_id, "")              # 句子缺失→空串


def build_prompt(sys_msg: str, evidence: str, claim: str) -> str:
    return (
        f"<s>[INST] {sys_msg}\n"
        f"Evidence: {evidence}\n"
        f"Claim: {claim}\n"
        "Question: Is the claim supported or refuted by the evidence?\n"
        "Answer:[/INST] "
    )


def prepare_dataset(keep_nei: bool = False,
                    cache_dir: str | None = None) -> Dataset:
    """把 FEVER train 拆分转换成 prompt+label 形式的 Dataset"""
    raw = load_dataset("fever", "v1.0", split="train", cache_dir=cache_dir)

    wiki = WikiCache(cache_dir)

    sys_msg = (
        "<<SYS>>\n"
        "You are a fact-checking assistant.\n"
        "Given EVIDENCE and a CLAIM, reply with exactly one token: "
        + ("SUPPORTED, REFUTED, or NOT ENOUGH INFO.\n"
           if keep_nei else
           "SUPPORTED or REFUTED.\n")
        + "Do not output anything else.\n"
        "<</SYS>>"
    )

    def convert(ex):
        label = ex["label"]
        if label == "NOT ENOUGH INFO" and not keep_nei:
            return None  # 过滤掉 NEI

        if label == "NOT ENOUGH INFO":
            ev_text = "No evidence provided."
            y = "NOT ENOUGH INFO"
        else:
            ev_text = wiki.sent(ex["evidence_id"], ex["evidence_sentence_id"])
            y = "SUPPORTED" if label == "SUPPORTS" else "REFUTED"

        prompt = build_prompt(sys_msg, ev_text, ex["claim"])
        return {"text": prompt + y + " </s>"}

    ds = raw.map(convert, remove_columns=raw.column_names)
    ds = ds.filter(lambda x: x is not None)
    return ds


# ------------------------- 训练器 --------------------------
def build_trainer(args) -> SFTTrainer:
    # ---- 登录 ----
    tok_str = args.token or os.getenv("HF_TOKEN")
    if tok_str:
        hf_login(tok_str)

    # ---- 模型 & 分词器 ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, add_eos_token=False, use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"  # 不影响 Llama
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype="auto"
    )
    lora_cfg = LoraConfig(r=8, lora_alpha=32,
                          target_modules=["q_proj", "v_proj"],
                          bias="none", lora_dropout=0.05)
    model = get_peft_model(base, lora_cfg)

    # ---- 数据 ----
    train_ds = prepare_dataset(args.keep_nei, args.cache_dir)

    # ---- 训练参数 ----
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=100,
        save_strategy="epoch",
        gradient_checkpointing=args.grad_ckpt,
    )

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        max_seq_length=512,
        args=targs,
        dataset_text_field="text",
    )


# ------------------------- CLI ----------------------------
def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--token", default=None, help="HF token (或用 HF_TOKEN 环境变量)")
    p.add_argument("--output_dir", default="./phaseA_llama3")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--grad_acc", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--keep_nei", action="store_true",
                   help="保留 NOT ENOUGH INFO，做三分类")
    p.add_argument("--cache_dir", default=None,
                   help="datasets 缓存目录；磁盘紧张时可指定大磁盘路径")
    return p.parse_args()


# ------------------------- MAIN ---------------------------
if __name__ == "__main__":
    cfg = parse_cli()
    trainer = build_trainer(cfg)
    trainer.train()
    trainer.save_model(cfg.output_dir)


