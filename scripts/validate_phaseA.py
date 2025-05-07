import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from phaseA_llama1b_fever import WikiCache # Assuming this is in PYTHONPATH or same dir (using 1b version for WikiCache as it's a utility)
import argparse

# --- CLI Argument Parsing ---
parser = argparse.ArgumentParser(description="Validate a Phase A model (LoRA or base).")
parser.add_argument("--base_model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                    help="Base model ID from Hugging Face Hub.")
parser.add_argument("--lora_path", type=str, default=None,
                    help="Path to the LoRA weights directory. If not provided or --no_lora is used, only base model is validated.")
parser.add_argument("--no_lora", action="store_true",
                    help="If set, do not load LoRA weights and validate the base model directly.")
parser.add_argument("--num_samples", type=int, default=1000,
                    help="Number of samples from the dev set to validate on.")
parser.add_argument("--batch_size", type=int, default=64,
                    help="Batch size for prediction.")
parser.add_argument("--max_length", type=int, default=512,
                    help="Max sequence length for tokenizer.")
parser.add_argument("--fixed_max_new_tokens", type=int, default=4,
                    help="Fixed number of max_new_tokens for generation. If 0 or negative, dynamic calculation is used.")
parser.add_argument("--token", type=str, default=None,
                    help="Hugging Face token for private models (or use HF_TOKEN env var).")

args = parser.parse_args()

BASE = args.base_model_id
LORA = args.lora_path if not args.no_lora and args.lora_path else None # Set LORA to None if --no_lora or lora_path is not given
NUM_SAMPLES_TO_VALIDATE = args.num_samples
BATCH_SIZE = args.batch_size
MAX_LENGTH = args.max_length
FIXED_MAX_NEW_TOKENS = args.fixed_max_new_tokens
HF_TOKEN = args.token

print(f"--- Configuration ---")
print(f"Base Model ID: {BASE}")
if LORA:
    print(f"LoRA Path: {LORA}")
else:
    print(f"LoRA Path: Not loading LoRA (using base model only).")
print(f"Samples to Validate: {NUM_SAMPLES_TO_VALIDATE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Max Sequence Length: {MAX_LENGTH}")
if FIXED_MAX_NEW_TOKENS > 0:
    print(f"Max New Tokens (Fixed): {FIXED_MAX_NEW_TOKENS}")
else:
    print(f"Max New Tokens: Dynamically calculated")
print(f"HF Token: {'Provided' if HF_TOKEN else 'Not Provided (using env or public model)'}")
print(f"---------------------\n")

# Initialize Tokenizer
try:
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=False, padding_side="left", token=HF_TOKEN)
except Exception as e:
    print(f"Error loading tokenizer for base model {BASE}: {e}")
    print("Please ensure the model ID is correct and you have internet access / necessary tokens.")
    exit()

if tok.pad_token is None:
    print("Setting pad_token to eos_token for tokenizer.")
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id

print("\n--- Tokenizer Details ---")
print(f"EOS token: '{tok.eos_token}', ID: {tok.eos_token_id}")
print(f"PAD token: '{tok.pad_token}', ID: {tok.pad_token_id}")
print(f"BOS token: '{tok.bos_token}', ID: {tok.bos_token_id}")
print(f"UNK token: '{tok.unk_token}', ID: {tok.unk_token_id}")
print(f"Tokenizer Vocabulary Size: {tok.vocab_size}")
print("\n--- Tokenizer Check (Informational) ---")
print(f"Tokenized 'SUPPORTED': {tok.tokenize('SUPPORTED')} -> IDs: {tok.encode('SUPPORTED', add_special_tokens=False)}")
print(f"Tokenized 'REFUTED': {tok.tokenize('REFUTED')} -> IDs: {tok.encode('REFUTED', add_special_tokens=False)}")
print("--- End Tokenizer Check ---\n")

# Load Model
try:
    model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", token=HF_TOKEN)
    print(f"Successfully loaded base model '{BASE}'.")
    if LORA:
        print(f"Attempting to load LoRA weights from '{LORA}'...")
        model = PeftModel.from_pretrained(model, LORA)
        print(f"Successfully loaded LoRA weights from '{LORA}'.")
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure the base model ID and LoRA path (if used) are correct and you have access (token if private).")
    exit()

# Prepare dataset
try:
    dev = load_dataset("fever", "v1.0", split="labelled_dev")
    dev = dev.filter(lambda r: r["label"] != "NOT ENOUGH INFO")
    wiki = WikiCache() # Assuming WikiCache does not depend on model size.
except Exception as e:
    print(f"Error loading dataset or WikiCache: {e}")
    exit()

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
def batch_predict_generate(prompts_list, batch_s=BATCH_SIZE): # Renamed prompts to prompts_list to avoid conflict
    preds = []
    print_once = True

    # Determine max_new_tokens
    if FIXED_MAX_NEW_TOKENS > 0:
        current_max_new_tokens = FIXED_MAX_NEW_TOKENS
        # print(f"Using FIXED max_new_tokens for generation: {current_max_new_tokens}\n") # Less verbose
    else:
        # Dynamic calculation
        # Add 1 for the EOS token itself if model generates it.
        # Add 1 more for a safety buffer.
        len_supported_tokens = len(tok.encode("SUPPORTED", add_special_tokens=False))
        len_refuted_tokens = len(tok.encode("REFUTED", add_special_tokens=False))
        current_max_new_tokens = max(len_supported_tokens, len_refuted_tokens, 1) + 2 # Ensure at least 1 + buffer
        print(f"Using DYNAMICALLY calculated max_new_tokens for generation: {current_max_new_tokens}\n")


    for i in range(0, len(prompts_list), batch_s):
        current_batch_prompts = prompts_list[i : i + batch_s]
        inputs = tok(
            current_batch_prompts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=current_max_new_tokens, # Use determined max_new_tokens
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            do_sample=False, # Ensure greedy
            temperature=1.0, # Default for greedy
            top_k=50,        # Default for greedy
            top_p=1.0        # Default for greedy
        )

        generated_texts = []
        for k_idx in range(generated_ids.shape[0]):
            # Ensure prompt_length_tokens doesn't exceed generated_ids length
            prompt_length_tokens = min(inputs.input_ids[k_idx].shape[0], generated_ids[k_idx].shape[0])
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
            first_word = normalized_gen_text.split()[0] if normalized_gen_text else ""
            
            if first_word.startswith("SUPPORTED"):
                preds.append("SUPPORTED")
            elif first_word.startswith("REFUTED"):
                preds.append("REFUTED")
            else: # Fallback for cases like "SUPPORTED." or if the first word is not the label
                if normalized_gen_text.startswith("SUPPORTED"):
                    preds.append("SUPPORTED")
                elif normalized_gen_text.startswith("REFUTED"):
                    preds.append("REFUTED")
                else:
                    preds.append("OTHER")
                    # Optional: Log these "OTHER" cases for debugging during the first batch
                    # if i == 0 and print_once is False: # After first batch prints are done
                    #     print(f"    -> Classified as OTHER (Raw: '{gen_text}', Normalized: '{normalized_gen_text}')")
    return preds

# Run prediction
pred = batch_predict_generate(prompts)

# Calculate accuracy
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
    f"\n--- Validation Results ---"
)
print(
    f"Clean dev Accuracy: {acc * 100:.2f}%   ({correct_predictions}/{total_processed})"
)
if other_predictions > 0:
    print(
        f"Number of 'OTHER' predictions (neither SUPPORTED nor REFUTED detected): {other_predictions} out of {total_processed}"
    )
print("\nValidation finished.")
