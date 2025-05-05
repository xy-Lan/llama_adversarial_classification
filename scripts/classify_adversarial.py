# scripts/classify_adversarial.py

# -*- coding: utf-8 -*-
"""Adversarial classification script – **outputs strictly `SUPPORTED` or `REFUTED`**.

基于用户现有脚本改写：
* prompt 最后加硬规则：只能回答两个单词。
* 自定义 `TwoLabelLimiter` `LogitsProcessor`，把除 SUPPORTED/REFUTED 之外的 logits 设为 −inf。
* `generate()` 最多 1 token，temperature=0。
* 解析时只取新 token；若首字母 “S” → SUPPORTED，否则 REFUTED。

可直接替换原脚本运行。
"""
import argparse
import os
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
)

############################################
# -------------  Data utils  ------------- #
############################################

def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def construct_prompts(df: pd.DataFrame):
    orig, adv, skipped = [], [], []
    for idx, row in df.iterrows():
        o, a = row["original_samples"], row["adversarial_samples"]
        if not isinstance(o, str) or not isinstance(a, str):
            skipped.append(idx); continue
        if "~" not in o or "~" not in a:
            skipped.append(idx); continue
        o_evi, o_clm = (s.strip() for s in o.split("~", 1))
        a_evi, a_clm = (s.strip() for s in a.split("~", 1))
        tmpl = (
            "Evidence: {e}\n"
            "Claim: {c}\n"
            "Question: Is this claim supported or refuted based on the evidence? "
            "Answer ONLY \"SUPPORTED\" or \"REFUTED\" (no other words)\n"
            "Answer:"
        )
        orig.append(tmpl.format(e=o_evi, c=o_clm))
        adv.append(tmpl.format(e=a_evi, c=a_clm))
    print(f"Total {len(df)} | Skipped {len(skipped)} | Valid {len(df)-len(skipped)}")
    return orig, adv, skipped, len(df)-len(skipped)

############################################
# -------------  Model utils ------------- #
############################################

class TwoLabelLimiter(LogitsProcessor):
    """Mask logits, leaving only ids in *allow_ids* active."""
    def __init__(self, allow_ids):
        self.allow = allow_ids
    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        mask[:, self.allow] = 0
        return scores + mask


def load_model(model_name: str, token: str | None):
    device_info = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading", model_name, "on", device_info)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # prefer bf16 else fp32
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=token,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_cfg,
    )
    model.eval()
    return tokenizer, model

############################################
# -------------  Inference  -------------- #
############################################

def classify(prompts, tokenizer, model, batch_size=32):
    device = next(model.parameters()).device

    allow_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("SUPPORTED", "REFUTED")]
    logits_proc = LogitsProcessorList([TwoLabelLimiter(allow_ids)])

    preds = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start+batch_size]
        enc = tokenizer(batch, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outs = model.generate(
                **enc,
                max_new_tokens=1,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=logits_proc,
            )
        inp_len = enc["input_ids"].shape[1]
        for o in outs:
            tok_id = o[inp_len].item()  # 只取新 token
            preds.append("SUPPORTED" if tok_id == allow_ids[0] else "REFUTED")
    return preds

############################################
# -------------  Metrics  ---------------- #
############################################

def evaluate(df, orig_preds, adv_preds, skipped, valid):
    sdf = df.drop(index=skipped).reset_index(drop=True)
    sdf["orig"], sdf["adv"] = orig_preds, adv_preds
    sdf["flip"] = sdf["orig"] != sdf["adv"]
    sdf["correct"] = sdf["orig"] == sdf["correctness"].str.upper()
    print("Clean acc:", f"{sdf['correct'].mean():.2%}")
    print("Flip rate:", f"{sdf['flip'].sum()/valid:.2%}")
    return sdf

############################################
# ---------------- main ------------------ #
############################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--token")
    ap.add_argument("--csv", default="./data/adversarial_dataset_corrected.csv")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    df = load_data(args.csv)
    orig, adv, skipped, valid = construct_prompts(df)
    tok, mdl = load_model(args.model, args.token)

    print("\nOriginal prompts …")
    o_preds = classify(orig, tok, mdl, args.batch)

    print("\nAdversarial prompts …")
    a_preds = classify(adv, tok, mdl, args.batch)

    evaluate(df, o_preds, a_preds, skipped, valid)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()



