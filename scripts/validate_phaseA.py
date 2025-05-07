import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from phaseA_llama1b_fever import WikiCache  # Assuming this is in PYTHONPATH or same dir
import argparse # Added for command-line arguments

# --- CLI Argument Parsing ---
parser = argparse.ArgumentParser(description="Validate a Phase A LoRA model.")
parser.add_argument("--base_model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                    help="Base model ID from Hugging Face Hub.")
parser.add_argument("--lora_path", type=str, default="./phaseA_llama3",
                    help="Path to the LoRA weights directory.")
parser.add_argument("--num_samples", type=int, default=1000,
                    help="Number of samples from the dev set to validate on.")
parser.add_argument("--batch_size", type=int, default=64,
                    help="Batch size for prediction.")
parser.add_argument("--max_length", type=int, default=512,
                    help="Max sequence length for tokenizer.")
parser.add_argument("--token", type=str, default=None,
                    help="Hugging Face token for private models (or use HF_TOKEN env var).")

args = parser.parse_args()

# Use parsed arguments
BASE = args.base_model_id
LORA = args.lora_path
NUM_SAMPLES_TO_VALIDATE = args.num_samples
BATCH_SIZE = args.batch_size
MAX_LENGTH = args.max_length
HF_TOKEN = args.token

print(f"--- Configuration ---")
print(f"Base Model ID: {BASE}")
print(f"LoRA Path: {LORA}")
print(f"Samples to Validate: {NUM_SAMPLES_TO_VALIDATE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Max Sequence Length: {MAX_LENGTH}")
print(f"HF Token: {'Provided' if HF_TOKEN else 'Not Provided (using env or public model)'}")
print(f"---------------------\n")

tok = AutoTokenizer.from_pretrained(BASE, use_fast=False, padding_side="left", token=HF_TOKEN)

if tok.pad_token is None:
    print("Setting pad_token to eos_token for tokenizer.")
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id

print("--- Tokenizer Check (Informational) ---")
print(f"Tokenized 'SUPPORTED': {tok.tokenize('SUPPORTED')}")
print(f"Tokenized 'REFUTED': {tok.tokenize('REFUTED')}")
print("--- End Tokenizer Check ---\n")

try:
    base_model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", token=HF_TOKEN)
    model = PeftModel.from_pretrained(base_model, LORA) 
    # model = model.merge_and_unload() # Keep it unmerged for PeftModel
    model.eval()
    print(f"Successfully loaded base model '{BASE}' and LoRA from '{LORA}'.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure the base model ID and LoRA path are correct and you have access (token if private).")
    exit()


# ---------- prepare dev set ----------
dev = load_dataset("fever", "v1.0", split="labelled_dev")
dev = dev.filter(lambda r: r["label"] != "NOT ENOUGH INFO")

wiki = WikiCache()

sys_msg_validation = (
    "<<SYS>>\\n"
    "You are a fact-checking assistant.\\n"
    "Given EVIDENCE and a CLAIM, reply with exactly one token: SUPPORTED or REFUTED.\\n"
    "Do not output anything else.\\n"
    "<</SYS>>"
)

def to_prompt(r):
    evid = wiki.sent(r["evidence_id"], r["evidence_sentence_id"])
    return (
        f"<s>[INST] {sys_msg_validation}\\n"
        f"Evidence: {evid}\\n"
        f"Claim: {r['claim']}\\n"
        "Question: Is this claim supported or refuted by the evidence?\\n"
        "Answer:[/INST] "
    )

dev_subset = dev.select(range(min(NUM_SAMPLES_TO_VALIDATE, len(dev))))
print(
    f"\n--- Running validation on the first {len(dev_subset)} samples of the dev set. ---\n"
)

prompts = list(map(to_prompt, dev_subset))
gold = ["SUPPORTED" if l == "SUPPORTS" else "REFUTED" for l in dev_subset["label"]]

@torch.no_grad()
def batch_predict_generate(prompts, bs=BATCH_SIZE): # Use BATCH_SIZE from args
    preds = []
    print_once = True

    max_len_supported = len(tok.tokenize("SUPPORTED")) + 2
    max_len_refuted = len(tok.tokenize("REFUTED")) + 2
    max_new_tokens = max(max_len_supported, max_len_refuted, 5)
    print(f"Using max_new_tokens for generation: {max_new_tokens}\n")

    for i in range(0, len(prompts), bs):
        current_batch_prompts = prompts[i : i + bs]

        inputs = tok(
            current_batch_prompts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH, # Use MAX_LENGTH from args
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )

        generated_texts = []
        for k_idx in range(generated_ids.shape[0]):
            prompt_length_tokens = inputs.input_ids[k_idx].shape[0]
            generated_part_ids = generated_ids[k_idx][prompt_length_tokens:]
            decoded_text = tok.decode(
                generated_part_ids, skip_special_tokens=True
            ).strip()
            generated_texts.append(decoded_text)

        if print_once and i == 0:
            print("\n--- First Batch Prompts (first 2 examples) ---")
            for k_idx in range(min(2, len(current_batch_prompts))):
                print(f"Example {k_idx+1} Prompt:\n{current_batch_prompts[k_idx]}\n")
            print("--- End First Batch Prompts ---")
            print("\n--- First Batch Generated Text (first 5 examples) ---")
            for k_idx in range(min(5, len(generated_texts))):
                print(f"Generated for Ex {k_idx+1}: '{generated_texts[k_idx]}'")
            print("--- End First Batch Generated Text ---\n")
            print_once = False

        for gen_text in generated_texts:
            normalized_gen_text = gen_text.upper().strip()
            if normalized_gen_text.startswith("SUPPORTED"):
                preds.append("SUPPORTED")
            elif normalized_gen_text.startswith("REFUTED"):
                preds.append("REFUTED")
            else:
                preds.append("OTHER")
    return preds

pred = batch_predict_generate(prompts)

correct_predictions = 0
other_predictions = 0
for p, g in zip(pred, gold):
    if p == g:
        correct_predictions += 1
    if p == "OTHER":
        other_predictions += 1

acc = correct_predictions / len(gold) if len(gold) > 0 else 0
total_processed = len(gold)

print(
    f"\nClean dev Accuracy: {acc * 100:.2f}%   ({correct_predictions}/{total_processed})"
)
if other_predictions > 0:
    print(
        f"Number of 'OTHER' predictions (neither SUPPORTED nor REFUTED detected): {other_predictions} out of {total_processed}"
    )

print("\nValidation finished.")