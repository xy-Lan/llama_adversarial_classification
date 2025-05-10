import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from phaseA_llama1b_fever import WikiCache # Assuming this is in PYTHONPATH or same dir
import argparse

# --- CLI Argument Parsing ---
# ... (parser setup remains the same as the full version previously provided)
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
parser.add_argument("--fixed_max_new_tokens", type=int, default=4, # Adjusted default, Llama3 tokens might be shorter
                    help="Fixed number of max_new_tokens for generation. If 0 or negative, dynamic calculation is used.")
parser.add_argument("--token", type=str, default=None,
                    help="Hugging Face token for private models (or use HF_TOKEN env var).")
args = parser.parse_args()

BASE = args.base_model_id
LORA = args.lora_path if not args.no_lora and args.lora_path else None
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
# ... (rest of the config print)
print(f"Max New Tokens (Fixed): {FIXED_MAX_NEW_TOKENS if FIXED_MAX_NEW_TOKENS > 0 else 'Dynamically calculated'}")
print(f"---------------------\n")


# Initialize Tokenizer
try:
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=False, token=HF_TOKEN) # Removed padding_side for now, will add if needed by Llama3 format
    # Llama 3 specific tokens. Add them if they don't exist, though for official Llama 3.2 models they should.
    # It's safer to rely on the tokenizer's pre-defined special tokens for Llama 3.
    # For Llama 3, typically no explicit pad_token is set; instead, eos_token is used for padding if needed.
    if tok.pad_token is None:
        print("Tokenizer does not have a pad_token. Setting pad_token to eos_token.")
        tok.pad_token = tok.eos_token # Common practice for Llama models
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left" # Important for batched generation

except Exception as e:
    print(f"Error loading tokenizer for base model {BASE}: {e}")
    exit()

print("\n--- Tokenizer Details (Llama 3 Focus) ---")
print(f"EOS token: '{tok.eos_token}', ID: {tok.eos_token_id}")
print(f"PAD token: '{tok.pad_token}', ID: {tok.pad_token_id}") # Should be same as EOS for Llama3 if not set
print(f"BOS token: '{tok.bos_token}', ID: {tok.bos_token_id}")
# Llama 3.2 uses specific header/eot tokens instead of a single system token for prompting.
# We will construct the prompt using these, rather than looking for a single <|system|> token.
print("--- End Tokenizer Details ---\n")


# Load Model (remains the same)
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
    exit()

# Prepare dataset (remains the same)
# ... (load_dataset, WikiCache)
try:
    dev = load_dataset("fever", "v1.0", split="labelled_dev")
    dev = dev.filter(lambda r: r["label"] != "NOT ENOUGH INFO")
    wiki = WikiCache() 
except Exception as e:
    print(f"Error loading dataset or WikiCache: {e}")
    exit()


# --- MODIFIED PROMPT FOR LLAMA 3.2 ---
# System message for Llama 3.2
llama3_system_message = (
    "You are a fact-checking assistant. "
    "Given EVIDENCE and a CLAIM, reply with exactly one token: SUPPORTED or REFUTED. "
    "Do not output anything else."
)

def to_llama3_prompt(r):
    evid = wiki.sent(r["evidence_id"], r["evidence_sentence_id"])
    # Constructing the prompt according to Llama 3.2 chat template
    # (Simplified version, official chat_template might be more complex with multiple turns)
    # The tokenizer usually has a .apply_chat_template() method which is preferred.
    # For direct construction for this specific task:
    prompt_parts = [
        f"<|begin_of_text|>", # Or tok.bos_token if it's <|begin_of_text|>
        f"<|start_header_id|>system<|end_header_id|>\n\n{llama3_system_message}<|eot_id|>",
        f"<|start_header_id|>user<|end_header_id|>\n\nEvidence: {evid}\nClaim: {r['claim']}\nQuestion: Is this claim supported or refuted by the evidence?<|eot_id|>",
        f"<|start_header_id|>assistant<|end_header_id|>\n\n" # Model generates after this
    ]
    return "".join(prompt_parts)

dev_subset = dev.select(range(min(NUM_SAMPLES_TO_VALIDATE, len(dev))))
print(
    f"\n--- Running validation on the first {len(dev_subset)} samples of the dev set (Llama 3.2 Prompt Format). ---\n"
)
prompts = list(map(to_llama3_prompt, dev_subset)) # Use new prompt function
gold = ["SUPPORTED" if l == "SUPPORTS" else "REFUTED" for l in dev_subset["label"]]


# batch_predict_generate function remains largely the same,
# but ensure it uses the new `prompts` and `tok`
@torch.no_grad()
def batch_predict_generate(prompts_list, batch_s=BATCH_SIZE):
    preds = []
    print_once = True

    if FIXED_MAX_NEW_TOKENS > 0:
        current_max_new_tokens = FIXED_MAX_NEW_TOKENS
    else:
        len_supported_tokens = len(tok.encode("SUPPORTED", add_special_tokens=False))
        len_refuted_tokens = len(tok.encode("REFUTED", add_special_tokens=False))
        current_max_new_tokens = max(len_supported_tokens, len_refuted_tokens, 1) + 2
        print(f"Using DYNAMICALLY calculated max_new_tokens for generation: {current_max_new_tokens}\n")
    if print_once: # Print max_new_tokens once
        print(f"Using max_new_tokens for generation: {current_max_new_tokens}\n")


    for i in range(0, len(prompts_list), batch_s):
        current_batch_prompts = prompts_list[i : i + batch_s]
        inputs = tok(
            current_batch_prompts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH, # Ensure this is adequate for Llama3 prompts
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=current_max_new_tokens,
            eos_token_id=tok.eos_token_id, # Llama3 uses <|eot_id|> or multiple specific end tokens
            pad_token_id=tok.pad_token_id,
            do_sample=False,
        )

        generated_texts = []
        for k_idx in range(generated_ids.shape[0]):
            prompt_length_tokens = min(inputs.input_ids[k_idx].shape[0], generated_ids[k_idx].shape[0])
            generated_part_ids = generated_ids[k_idx][prompt_length_tokens:]
            # For Llama 3, it's crucial to handle its specific EOS tokens correctly during decoding.
            # `skip_special_tokens=True` should handle <|eot_id|> etc.
            decoded_text = tok.decode(
                generated_part_ids, skip_special_tokens=True
            ).strip()
            generated_texts.append(decoded_text)

        if print_once and i == 0:
            print("\n--- First Batch Prompts (Llama 3.2 format, first example) ---")
            if current_batch_prompts: print(f"Example 1 Prompt:\n{current_batch_prompts[0]}\n")
            print("--- End First Batch Prompts ---")
            print("\n--- First Batch Generated Text (first 5 examples) ---")
            for k_idx in range(min(5, len(generated_texts))):
                print(f"Generated for Ex {k_idx+1}: '{generated_texts[k_idx]}'")
            print("--- End First Batch Generated Text ---\n")
            print_once = False # Set false after first batch print

        for gen_text in generated_texts:
            normalized_gen_text = gen_text.upper().strip()
            first_word = normalized_gen_text.split()[0] if normalized_gen_text else ""
            
            if first_word.startswith("SUPPORTED"):
                preds.append("SUPPORTED")
            elif first_word.startswith("REFUTED"):
                preds.append("REFUTED")
            else:
                if normalized_gen_text.startswith("SUPPORTED"): # Fallback
                    preds.append("SUPPORTED")
                elif normalized_gen_text.startswith("REFUTED"): # Fallback
                    preds.append("REFUTED")
                else:
                    preds.append("OTHER")
    return preds

# ... (rest of the script: run prediction, calculate accuracy, print results)
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
