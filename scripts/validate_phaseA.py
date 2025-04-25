import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "meta-llama/Llama-3.2-1B-Instruct"          # 换成 1B / 8B 亦可
LORA = "./phaseA_llama3"                            # 你的输出目录

tok = AutoTokenizer.from_pretrained(BASE, use_fast=False)
base = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto")
model = PeftModel.from_pretrained(base, LORA).merge_and_unload()
model.eval()

# ---------- prepare dev set ----------
dev = load_dataset("fever", "v1.0", split="validation")
dev = dev.filter(lambda r: r["label"] != "NOT ENOUGH INFO")

def to_prompt(r):
    evidence = r["evidence"]
    return (
        f"Evidence: {evidence}\n"
        f"Claim: {r['claim']}\n"
        "Question: Is this claim supported or refuted by the evidence?\n"
        "Answer:"
    )
prompts = list(map(to_prompt, dev))
gold    = ["SUPPORTED" if l=="SUPPORTS" else "REFUTED" for l in dev["label"]]

# ---------- batched prediction ----------
@torch.no_grad()
def predict(ps, bs=16):
    preds = []
    for i in range(0, len(ps), bs):
        sub = tok(ps[i:i+bs], padding=True,
                  truncation=True, max_length=512,
                  return_tensors="pt").to(model.device)
        out = model.generate(**sub, max_new_tokens=1)
        preds += [tok.decode(o, skip_special_tokens=True).split()[0].upper()
                  for o in out]
    return preds

pred = predict(prompts)
acc  = sum(p==g for p,g in zip(pred, gold)) / len(gold)
print(f"Clean dev Accuracy: {acc*100:.2f}%   ({acc*len(gold)}/{len(gold)})")
