import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from phaseA_llama1b_fever import WikiCache # Assuming this is in PYTHONPATH or same dir

BASE = "meta-llama/Llama-3.2-1B-Instruct"
LORA = "./phaseA_llama3"

tok = AutoTokenizer.from_pretrained(BASE, use_fast=False)
tok.pad_token = tok.eos_token

# <DEBUG> Check if SUPPORTED and REFUTED are single tokens
print("--- Tokenizer Check ---")
tokenized_supported = tok.tokenize('SUPPORTED')
id_supported = tok.convert_tokens_to_ids('SUPPORTED') # Might be a list if multi-token
if isinstance(id_supported, list): # Handling if it's multi-token
    id_supported = id_supported[0] if len(id_supported) > 0 else -1 # Or handle error

tokenized_refuted = tok.tokenize('REFUTED')
id_refuted = tok.convert_tokens_to_ids('REFUTED') # Might be a list if multi-token
if isinstance(id_refuted, list): # Handling if it's multi-token
    id_refuted = id_refuted[0] if len(id_refuted) > 0 else -1 # Or handle error


print(f"Tokenized 'SUPPORTED': {tokenized_supported}")
print(f"ID for 'SUPPORTED' (first token if multi): {id_supported}")
print(f"Tokenized 'REFUTED': {tokenized_refuted}")
print(f"ID for 'REFUTED' (first token if multi): {id_refuted}")
print("--- End Tokenizer Check ---")


base = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto")
model = PeftModel.from_pretrained(base, LORA).merge_and_unload()
model.eval()

# ---------- prepare dev set ----------
dev = load_dataset("fever", "v1.0", split="labelled_dev")
dev = dev.filter(lambda r: r["label"] != "NOT ENOUGH INFO")

wiki = WikiCache()

# Ensure sys_msg_validation is defined as discussed
sys_msg_validation = (
    "<<SYS>>\n"
    "You are a fact-checking assistant.\n"
    "Given EVIDENCE and a CLAIM, reply with exactly one token: "
    "SUPPORTED or REFUTED.\n"
    "Do not output anything else.\n"
    "<</SYS>>"
)

def to_prompt(r):
    evid = wiki.sent(r["evidence_id"], r["evidence_sentence_id"])
    return (
        f"<s>[INST] {sys_msg_validation}\n"
        f"Evidence: {evid}\n"
        f"Claim: {r['claim']}\n"
        "Question: Is this claim supported or refuted by the evidence?\n"
        "Answer:[/INST] "
    )

prompts = list(map(to_prompt, dev))
gold = ["SUPPORTED" if l == "SUPPORTS" else "REFUTED" for l in dev["label"]]


# ---------- batched prediction ----------
@torch.no_grad()
def batch_predict(prompts, bs=64):
    preds = []
    print_once = True
    for i in range(0, len(prompts), bs):
        current_batch_prompts = prompts[i : i + bs]
        if print_once and i == 0:
            print("\n--- First Batch Prompts (first 2 examples) ---")
            for k in range(min(2, len(current_batch_prompts))):
                print(f"Example {k+1}:\n{current_batch_prompts[k]}\n")
            print("--- End First Batch Prompts ---")

        sub = tok(
            current_batch_prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**sub)
            logits = outputs.logits[:, -1]
        
        idx = logits.argmax(-1).tolist()

        if print_once and i == 0:
            print("\n--- First Batch Predictions (Raw) ---")
            print(f"Predicted Token IDs (first 5): {idx[:5]}")
            try:
                predicted_tokens = tok.convert_ids_to_tokens(idx)
                print(f"Predicted Tokens (first 5): {predicted_tokens[:5]}")
            except Exception as e:
                print(f"Error converting IDs to tokens: {e}. IDs: {idx[:5]}")

            # Optional: Print logits for the first example's prediction
            if len(idx) > 0 and len(logits) > 0 : # Check if logits and idx are not empty
                 print(f"Logits for first example's prediction (Top 5):")
                 top_k_logits, top_k_ids = torch.topk(logits[0], 5)
                 for logit_val, token_id_val in zip(top_k_logits, top_k_ids):
                    print(f"  Token: {tok.convert_ids_to_tokens([token_id_val.item()])[0]}, Logit: {logit_val.item()}")

            print("--- End First Batch Predictions (Raw) ---")
            print_once = False

        # Make sure id_supported and id_refuted are integers for comparison
        current_id_supported = id_supported
        if isinstance(id_supported, list): current_id_supported = id_supported[0]
        
        current_id_refuted = id_refuted # Not used if we default to REFUTED

        preds += [
            "SUPPORTED" if j == current_id_supported else "REFUTED"
            for j in idx
        ]
    return preds

pred = batch_predict(prompts)
acc = sum(p == g for p, g in zip(pred, gold)) / len(gold)
print(f"Clean dev Accuracy: {acc * 100:.2f}%   ({int(acc * len(gold))}/{len(gold)})")