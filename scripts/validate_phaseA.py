import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from phaseA_llama1b_fever import WikiCache  # Assuming this is in PYTHONPATH or same dir

BASE = "meta-llama/Llama-3.2-1B-Instruct"
LORA = "./phaseA_llama3"  # 你的输出目录

tok = AutoTokenizer.from_pretrained(BASE, use_fast=False, padding_side="left")

# It's crucial that pad_token is set, especially for generate
if tok.pad_token is None:
    print("Setting pad_token to eos_token for tokenizer.")
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id  # Ensure ID is also set

# Informational print about how tokenizer handles these words
print("--- Tokenizer Check (Informational) ---")
print(f"Tokenized 'SUPPORTED': {tok.tokenize('SUPPORTED')}")
print(f"Tokenized 'REFUTED': {tok.tokenize('REFUTED')}")
print("--- End Tokenizer Check ---")

base = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto")
model = PeftModel.from_pretrained(base, LORA).merge_and_unload()
model.eval()

# ---------- prepare dev set ----------
dev = load_dataset("fever", "v1.0", split="labelled_dev")
dev = dev.filter(lambda r: r["label"] != "NOT ENOUGH INFO")

wiki = WikiCache()

sys_msg_validation = (
    "<<SYS>>\\n"
    "You are a fact-checking assistant.\\n"
    # Instruction below is misleading for multi-token, but kept for consistency with training prompt
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
        "Answer:[/INST] "  # Model will generate tokens after this
    )


# Take only the first 1000 samples from the dev set
num_samples_to_validate = 1000
dev_subset = dev.select(range(min(num_samples_to_validate, len(dev))))
print(
    f"\n--- Running validation on the first {len(dev_subset)} samples of the dev set. ---\n"
)

prompts = list(map(to_prompt, dev_subset))
gold = ["SUPPORTED" if l == "SUPPORTS" else "REFUTED" for l in dev_subset["label"]]


# ---------- batched prediction using model.generate() ----------
@torch.no_grad()
def batch_predict_generate(prompts, bs=64):
    preds = []
    print_once = True

    # Determine max_new_tokens: longest of "SUPPORTED" and "REFUTED" token sequences + a small buffer
    max_len_supported = (
        len(tok.tokenize("SUPPORTED")) + 2
    )  # Buffer for EOS or other variations
    max_len_refuted = len(tok.tokenize("REFUTED")) + 2  # Buffer for EOS
    # Ensure a minimum number of tokens can be generated, e.g., if words are very short
    max_new_tokens = max(
        max_len_supported, max_len_refuted, 5
    )  # Min 5 tokens for safety margin
    print(f"Using max_new_tokens for generation: {max_new_tokens}")

    for i in range(0, len(prompts), bs):
        current_batch_prompts = prompts[i : i + bs]

        inputs = tok(
            current_batch_prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            # early_stopping=True, # Consider enabling if generation often stops early with EOS
            # num_beams=1, # Default is greedy search
            # do_sample=False # Default is greedy search
        )

        # Decode only the generated part
        generated_texts = []
        for k in range(generated_ids.shape[0]):
            prompt_length_tokens = inputs.input_ids[k].shape[0]
            # Slice generated_ids to get only the newly generated tokens
            generated_part_ids = generated_ids[k][prompt_length_tokens:]
            decoded_text = tok.decode(
                generated_part_ids, skip_special_tokens=True
            ).strip()
            generated_texts.append(decoded_text)

        if print_once and i == 0:
            print("\\n--- First Batch Prompts (first 2 examples) ---")
            for k in range(min(2, len(current_batch_prompts))):
                print(f"Example {k+1} Prompt:\\n{current_batch_prompts[k]}\\n")
            print("--- End First Batch Prompts ---")
            print("\\n--- First Batch Generated Text (first 5 examples) ---")
            for k in range(min(5, len(generated_texts))):
                print(f"Generated for Ex {k+1}: '{generated_texts[k]}'")
            print("--- End First Batch Generated Text ---")
            print_once = False

        for gen_text in generated_texts:
            # Normalize generated text for robust comparison
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
    if p == "OTHER":  # Count how many were not classified as SUPPORTED or REFUTED
        other_predictions += 1

acc = correct_predictions / len(gold) if len(gold) > 0 else 0
total_processed = len(gold)

print(
    f"Clean dev Accuracy: {acc * 100:.2f}%   ({correct_predictions}/{total_processed})"
)
if other_predictions > 0:
    print(
        f"Number of 'OTHER' predictions (neither SUPPORTED nor REFUTED detected): {other_predictions} out of {total_processed}"
    )
