import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from phaseA_llama1b_fever import WikiCache

BASE = "meta-llama/Llama-3.2-1B-Instruct"  # 换成 1B / 8B 亦可
LORA = "./phaseA_llama3"  # 你的输出目录

tok = AutoTokenizer.from_pretrained(BASE, use_fast=False)
base = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto")
model = PeftModel.from_pretrained(base, LORA).merge_and_unload()
model.eval()

# ---------- prepare dev set ----------
dev = load_dataset("fever", "v1.0", split="labelled_dev")
dev = dev.filter(lambda r: r["label"] != "NOT ENOUGH INFO")

wiki = WikiCache()


def to_prompt(r):
    evid = wiki.sent(r["evidence_id"], r["evidence_sentence_id"])
    return (
        f"Evidence: {evid}\n"
        f"Claim: {r['claim']}\n"
        "Question: Is this claim supported or refuted by the evidence?\n"
        "Answer:"
    )


prompts = list(map(to_prompt, dev))
gold = ["SUPPORTED" if l == "SUPPORTS" else "REFUTED" for l in dev["label"]]


# ---------- batched prediction ----------
@torch.no_grad()
def batch_predict(prompts, bs=64):
    preds = []
    for i in range(0, len(prompts), bs):
        sub = tok(prompts[i:i + bs], padding=True, truncation=True,
                  max_length=512, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**sub).logits[:, -1]  # 不用 generate
        idx = logits.argmax(-1).tolist()
        preds += ["SUPPORTED" if j == tok.convert_tokens_to_ids('SUPPORTED') else "REFUTED"
                  for j in idx]
    return preds


pred = batch_predict(prompts)
acc = sum(p == g for p, g in zip(pred, gold)) / len(gold)
print(f"Clean dev Accuracy: {acc * 100:.2f}%   ({acc * len(gold)}/{len(gold)})")
