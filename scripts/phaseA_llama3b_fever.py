#!/usr/bin/env python3
# phaseA_llama3b_fever.py  —— 微调 Llama-3-3B-Instruct 在 FEVER 二分类 (Llama 3.2 Prompt Format)

import os, argparse
from typing import Dict
from datasets import load_dataset, Dataset
from huggingface_hub import login as hf_login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ------------------------- Wiki 缓存 -------------------------
class WikiCache:
    def __init__(self, cache_dir=None):
        wiki = load_dataset("fever", "wiki_pages", cache_dir=cache_dir, trust_remote_code=True)["wikipedia_pages"]
        self._idx = {row["id"]: i for i, row in enumerate(wiki)}
        self._data, self._cache = wiki, {}

    def sent(self, page_id, sent_id: int) -> str:
        key = str(page_id)
        if key not in self._cache:
            if key not in self._idx:
                return ""
            rec = self._data[self._idx[key]]
            self._cache[key] = {
                int(line.split("\t", 1)[0]): line.split("\t", 1)[1]
                for line in rec["lines"].split("\n") if line
            }
        return self._cache[key].get(sent_id, "")

# --- MODIFIED PROMPT FOR LLAMA 3.2 TRAINING ---
llama3_system_message_train = (
    "You are a fact-checking assistant. "
    "Given EVIDENCE and a CLAIM, reply with SUPPORTED or REFUTED. " # Simplified instruction
    "Do not output anything else."
)
llama3_system_message_train_nei = (
    "You are a fact-checking assistant. "
    "Given EVIDENCE and a CLAIM, reply with SUPPORTED, REFUTED, or NOT ENOUGH INFO. "
    "Do not output anything else."
)

def build_prompt_llama3(sys_msg_content: str, evidence: str, claim: str) -> str:
    prompt_parts = [
        f"<|begin_of_text|>",
        f"<|start_header_id|>system<|end_header_id|>\n\n{sys_msg_content}<|eot_id|>",
        f"<|start_header_id|>user<|end_header_id|>\n\nEvidence: {evidence}\nClaim: {claim}\nQuestion: Is this claim supported or refuted by the evidence?<|eot_id|>",
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    ]
    return "".join(prompt_parts)

# MODIFIED: prepare_dataset now takes tokenizer as an argument
def prepare_dataset(tokenizer: AutoTokenizer, keep_nei=False, cache_dir=None) -> Dataset:
    raw  = load_dataset("fever", "v1.0", split="train", cache_dir=cache_dir, trust_remote_code=True)
    wiki = WikiCache(cache_dir)
        
    current_sys_msg = llama3_system_message_train_nei if keep_nei else llama3_system_message_train

    eos_token_to_use = tokenizer.eos_token
    if eos_token_to_use is None: # Should not happen with Llama 3 tokenizers
        print("Warning: tokenizer.eos_token is None. This is unexpected for Llama 3. Manually setting to <|eot_id|>.")
        eos_token_to_use = "<|eot_id|>"
    
    print(f"prepare_dataset will use EOS token: '{eos_token_to_use}' for constructing 'text' field.")

    def convert(ex):
        lab = ex["label"]
        if lab == "NOT ENOUGH INFO" and not keep_nei:
            return None
            
        y_text = "" 
        if lab == "NOT ENOUGH INFO":
            ev = "No evidence provided." 
            y_text = "NOT ENOUGH INFO"
        elif lab == "SUPPORTS":
            ev = wiki.sent(ex["evidence_id"], ex["evidence_sentence_id"])
            y_text = "SUPPORTED"
        elif lab == "REFUTES":
            ev = wiki.sent(ex["evidence_id"], ex["evidence_sentence_id"])
            y_text = "REFUTED"
        else: 
            return None
                
        prompt_part = build_prompt_llama3(current_sys_msg, ev, ex["claim"])
        full_text = prompt_part + y_text + eos_token_to_use # Use dynamically obtained EOS token
        return {"text": full_text}

    ds = raw.map(convert, remove_columns=raw.column_names)
    return ds.filter(lambda x: x is not None)

# ------------------------- Trainer --------------------------
def build_trainer(cfg) -> SFTTrainer:
    if cfg.token or os.getenv("HF_TOKEN"):
        hf_login(token=cfg.token or os.getenv("HF_TOKEN")) # Use token argument

    # Load tokenizer
    # For SFTTrainer with dataset_text_field, add_eos_token=False is usually correct
    # if the text field already contains the EOS token.
    tok = AutoTokenizer.from_pretrained(cfg.model_id, add_eos_token=False, use_fast=False)
    
    if tok.pad_token is None:
        print("Tokenizer does not have a pad_token. Setting pad_token to eos_token.")
        tok.pad_token = tok.eos_token # Common practice for Llama models
        tok.pad_token_id = tok.eos_token_id # Ensure ID is also set
    
    tok.padding_side = "left" # MODIFIED: Crucial for batched Causal LM

    print(f"\n--- Trainer Tokenizer Details ---")
    print(f"EOS token used by tokenizer: '{tok.eos_token}' (ID: {tok.eos_token_id})")
    print(f"PAD token used by tokenizer: '{tok.pad_token}' (ID: {tok.pad_token_id}), Padding Side: {tok.padding_side}")
    print(f"---------------------------------\n")

    # Load base model
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, device_map="auto", torch_dtype="auto")
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], # Consider adding "k_proj", "o_proj" for Llama 3 if issues persist
        bias="none", 
        lora_dropout=0.05
    )
    model = get_peft_model(base, lora_config)
    model.print_trainable_parameters() # Good to check

    # Prepare dataset using the loaded tokenizer
    train_ds = prepare_dataset(tok, cfg.keep_nei, cfg.cache_dir) # MODIFIED: Pass tokenizer
    
    # Training Arguments
    targs = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch,
        gradient_accumulation_steps=cfg.grad_acc,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=cfg.fp16, 
        bf16=cfg.bf16,
        logging_steps=100, 
        save_strategy="epoch",
        gradient_checkpointing=cfg.grad_ckpt,
        logging_dir=f"{cfg.output_dir}/runs", # Enable TensorBoard logging
    )

    return SFTTrainer(
        model=model, 
        tokenizer=tok, 
        train_dataset=train_ds,
        max_seq_length=512, # Ensure this is enough for Llama 3 prompts + answer + EOS
        dataset_text_field="text", 
        args=targs
    )

# ------------------------- CLI ----------------------------
def cli():
    ap = argparse.ArgumentParser(description="Fine-tune Llama-3.2-3B for FEVER fact-checking (Llama 3.2 Prompt Format)")
    ap.add_argument("--model_id", default="meta-llama/Llama-3.2-3B-Instruct", help="Base model ID.")
    ap.add_argument("--token", default=None, help="Hugging Face token (or use HF_TOKEN env var).")
    ap.add_argument("--output_dir", default="./phaseA_llama3B_L3prompt", # Suggest new dir for new prompt format
                    help="Directory to save LoRA weights and logs.")
    ap.add_argument("--batch", type=int, default=16, help="Per device batch size.")
    ap.add_argument("--grad_acc", type=int, default=2, help="Gradient accumulation steps.")
    ap.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    ap.add_argument("--grad_ckpt", action="store_true", help="Enable gradient checkpointing.")
    ap.add_argument("--fp16", action="store_true", help="Enable FP16 training.")
    ap.add_argument("--bf16", action="store_true", help="Enable BF16 training.")
    ap.add_argument("--keep_nei", action="store_true", help="Include NOT ENOUGH INFO (3-way classification).")
    ap.add_argument("--cache_dir", default=None, help="Directory for Hugging Face datasets cache.")
    return ap.parse_args()

# ------------------------- MAIN ---------------------------
if __name__ == "__main__":
    cfg = cli()
    print(f"--- Training Configuration ---")
    for arg, value in vars(cfg).items():
        print(f"{arg}: {value}")
    print(f"------------------------------\n")

    # Create output directory if it doesn't exist
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)
        print(f"Created output directory: {cfg.output_dir}")

    trainer = build_trainer(cfg)
    
    print("\nStarting training...")
    trainer.train()
    print("Training finished.")
    
    print(f"\nSaving model to {cfg.output_dir}...")
    trainer.save_model(cfg.output_dir) # Saves LoRA adapter
    print("Model saved.")

