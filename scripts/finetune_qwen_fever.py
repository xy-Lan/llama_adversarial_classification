#!/usr/bin/env python3
# scripts/finetune_qwen_fever.py
# Fine-tune Qwen on FEVER, strictly replicating data processing from phaseA_llama3b_fever.py

import os
import argparse
from datasets import load_dataset, Dataset
from huggingface_hub import login as hf_login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch
import logging

# --- Start: Set Cache Directories ---
CACHE_DIR_TARGET = "/mnt/parscratch/users/acc22xl/huggingface"
os.environ["HF_HOME"] = CACHE_DIR_TARGET
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR_TARGET
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR_TARGET
os.environ["HF_METRICS_CACHE"] = CACHE_DIR_TARGET
# --- End: Set Cache Directories ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------- Wiki 缓存 (from phaseA_llama3b_fever.py) -------------------------
class WikiCache:
    def __init__(self, cache_dir=None):
        # Trust remote code for fever dataset loading
        wiki = load_dataset("fever", "wiki_pages", cache_dir=cache_dir, trust_remote_code=True)["wikipedia_pages"]
        self._idx = {row["id"]: i for i, row in enumerate(wiki)}
        self._data, self._cache = wiki, {}
        logging.info(f"WikiCache (Llama-style) initialized with {len(self._idx)} pages from cache_dir: {cache_dir or 'default'}.")

    def sent(self, page_id, sent_id: int) -> str:
        key = str(page_id) # page_id from dataset could be non-string, Llama script implies it could be.
        if key not in self._cache:
            if key not in self._idx:
                # logging.warning(f"WikiCache (Llama-style): Page ID '{key}' not found in index.")
                return "" # Return empty string if page_id is not found
            try:
                rec = self._data[self._idx[key]]
                lines_data = rec.get('lines')
                if lines_data is None or not isinstance(lines_data, str):
                    logging.warning(f"WikiCache (Llama-style): Page ID '{key}' has no 'lines' string attribute. Lines: {lines_data}")
                    return ""

                self._cache[key] = {
                    # Llama script: int(line.split("\t", 1)[0]): line.split("\t", 1)[1] for line in rec["lines"].split("\n") if line
                    # This assumes tab exists and split yields 2 parts.
                    # More robustly (pre Python 3.8 compatible):
                    int(line.split("\t", 1)[0]): line.split("\t", 1)[1]
                    for line in lines_data.split("\n")
                    if line and "\t" in line and len(line.split("\t", 1)) == 2
                }
            except Exception as e:
                logging.error(f"WikiCache (Llama-style): Error processing page ID '{key}' for caching: {e}", exc_info=True)
                return "" 
        
        # Llama script's get: self._cache[key].get(sent_id, "")
        # Ensure sent_id is an int for lookup
        try:
            sent_id_int = int(sent_id)
        except (ValueError, TypeError):
            # logging.warning(f"WikiCache (Llama-style): sent_id '{sent_id}' could not be converted to int for page_id '{key}'.")
            return ""

        sentence = self._cache.get(key, {}).get(sent_id_int, "")
        # if not sentence:
            # logging.warning(f"WikiCache (Llama-style): Sentence ID '{sent_id_int}' not found for page ID '{key}'.")
        return sentence

# --- Qwen Prompt Formatting (Retained from Qwen script) ---
QWEN_SYSTEM_MESSAGE_TRAIN = (
    "You are a fact-checking assistant. "
    "Given EVIDENCE and a CLAIM, reply with SUPPORTED or REFUTED. "
    "Do not output anything else."
)
QWEN_SYSTEM_MESSAGE_TRAIN_NEI = (
    "You are a fact-checking assistant. "
    "Given EVIDENCE and a CLAIM, reply with SUPPORTED, REFUTED, or NOT ENOUGH INFO. "
    "Do not output anything else."
)

def format_qwen_chat_for_sft(tokenizer, system_message_content: str, evidence: str, claim: str, label_text: str) -> str:
    messages = [
        {"role": "system", "content": system_message_content},
        {"role": "user", "content": f"Evidence: {evidence}\nClaim: {claim}\nQuestion: Is this claim supported or refuted by the evidence?"},
        {"role": "assistant", "content": label_text}
    ]
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False # Important for SFT
        )
        return formatted_prompt
    except Exception as e:
        logging.error(f"Error applying Qwen chat template: {e}. Messages: {messages}", exc_info=True)
        return None

# --- Dataset Preparation (Strictly Llama Script Logic Adapted for Qwen) ---
def prepare_dataset(tokenizer: AutoTokenizer, keep_nei=False, cache_dir=None) -> Dataset:
    effective_cache_dir = cache_dir or CACHE_DIR_TARGET
    logging.info(f"prepare_dataset (Strict Llama Logic for Qwen) loading FEVER v1.0 from cache_dir: {effective_cache_dir}")
    
    try:
        # Trust remote code for fever dataset loading, as in Llama script
        raw = load_dataset("fever", "v1.0", split="train", cache_dir=effective_cache_dir, trust_remote_code=True)
        logging.info(f"DEBUG (Strict Llama Logic for Qwen): Loaded 'raw' dataset. Length: {len(raw)}")
    except Exception as e_load:
        logging.error(f"CRITICAL_ERROR (Strict Llama Logic for Qwen): Failed to load dataset. Error: {e_load}", exc_info=True)
        raise

    wiki = WikiCache(cache_dir=effective_cache_dir)
    current_sys_msg = QWEN_SYSTEM_MESSAGE_TRAIN_NEI if keep_nei else QWEN_SYSTEM_MESSAGE_TRAIN
    
    # For tracking, not in Llama script but useful for debugging
    skipped_counts = {"NEI_SKIPPED": 0, "UNKNOWN_LABEL": 0, "KEY_ERROR_EVIDENCE": 0, "TEMPLATE_ERROR": 0, "EMPTY_EVIDENCE_AFTER_LOOKUP": 0}

    def convert(ex):
        lab = ex["label"]
        
        if lab == "NOT ENOUGH INFO" and not keep_nei:
            skipped_counts["NEI_SKIPPED"] += 1
            return None # Llama script returns None
            
        y_text = "" 
        evidence_text = "" # Llama script's `ev`

        if lab == "NOT ENOUGH INFO": # keep_nei must be True here
            evidence_text = "No evidence provided." # Llama script uses this for NEI
            y_text = "NOT ENOUGH INFO"
        elif lab == "SUPPORTS" or lab == "REFUTES":
            try:
                # Directly access ex["evidence_id"] and ex["evidence_sentence_id"] as per Llama script
                # This is the critical part that assumes these keys exist in `fever, v1.0` items
                page_id = ex["evidence_id"] 
                sent_id = ex["evidence_sentence_id"] # Llama's WikiCache expects sent_id as int.
                evidence_text = wiki.sent(page_id, sent_id) # sent_id will be int-coerced in wiki.sent
                
                # Llama script does not explicitly check if evidence_text is empty here and return None.
                # It proceeds to build prompt. If evidence_text is "", it gets passed as empty.
                if not evidence_text: # Add a counter for this specific case
                    skipped_counts["EMPTY_EVIDENCE_AFTER_LOOKUP"] +=1
                    # Llama script would proceed with empty evidence. We follow that.
                    # logging.warning(f"Sample {ex.get('id')} ({lab}): Evidence lookup for page '{page_id}', sent '{sent_id}' returned empty. Proceeding.")

            except KeyError as ke:
                logging.warning(f"Sample {ex.get('id')} ({lab}): KeyError accessing 'evidence_id' or 'evidence_sentence_id': {ke}. Skipping.")
                skipped_counts["KEY_ERROR_EVIDENCE"] += 1
                return None # If keys are missing, this sample cannot be processed as per Llama logic.
            
            y_text = "SUPPORTED" if lab == "SUPPORTS" else "REFUTED"
        else: 
            logging.warning(f"Sample {ex.get('id')}: Unknown label '{lab}'. Skipping.")
            skipped_counts["UNKNOWN_LABEL"] += 1
            return None # Llama script returns None for other labels
        
        # Use Qwen's prompt formatting. Llama script builds its own prompt and appends EOS.
        # Qwen's apply_chat_template should handle EOS if add_generation_prompt=False.
        full_text = format_qwen_chat_for_sft(tokenizer, current_sys_msg, evidence_text, ex["claim"], y_text)
        
        if full_text is None: # Qwen formatting failed
            skipped_counts["TEMPLATE_ERROR"] += 1
            logging.warning(f"Sample {ex.get('id')}: format_qwen_chat_for_sft returned None. Skipping.")
            return None # If prompt formatting fails, skip.
            
        return {"text": full_text}

    # Llama script: ds = raw.map(convert, remove_columns=raw.column_names)
    # Not specifying num_proc, defaults to 1.
    processed_ds = raw.map(convert, remove_columns=raw.column_names) 
    
    logging.info(f"DEBUG (Strict Llama Logic for Qwen): Finished raw.map(convert). Initial mapped size: {len(processed_ds)}")
    for key, value in skipped_counts.items():
        if value > 0: logging.info(f"  Skipped in 'convert' due to '{key}': {value}")

    # Llama script: return ds.filter(lambda x: x is not None)
    final_ds = processed_ds.filter(lambda x: x is not None)
    logging.info(f"DEBUG (Strict Llama Logic for Qwen): Finished filtering None values. Final dataset size: {len(final_ds)}")
    
    if len(final_ds) == 0 and len(raw) > 0:
        logging.warning("WARNING (Strict Llama Logic for Qwen): The final_dataset is empty after processing!")

    return final_ds

# ------------------------- Trainer (Qwen Specific) --------------------------
def build_trainer(cfg) -> SFTTrainer:
    if cfg.token or os.getenv("HF_TOKEN"):
        logging.info(f"Logging into Hugging Face Hub with token.")
        hf_login(token=cfg.token or os.getenv("HF_TOKEN"))

    logging.info(f"Loading Qwen tokenizer for: {cfg.model_id} from cache_dir: {cfg.cache_dir}")
    tok = AutoTokenizer.from_pretrained(
        cfg.model_id,
        use_fast=True,
        trust_remote_code=True,
        cache_dir=cfg.cache_dir,
    )
    
    if tok.pad_token is None:
        logging.info("Qwen tokenizer does not have a pad_token. Setting pad_token to eos_token.")
        tok.pad_token = tok.eos_token 
        tok.pad_token_id = tok.eos_token_id
    
    tok.padding_side = "left" 

    logging.info(f"--- Qwen Trainer Tokenizer Details ---")
    logging.info(f"EOS token: '{tok.eos_token}' (ID: {tok.eos_token_id})")
    logging.info(f"PAD token: '{tok.pad_token}' (ID: {tok.pad_token_id}), Padding Side: {tok.padding_side}")
    logging.info(f"---------------------------------")

    logging.info(f"Loading Qwen base model: {cfg.model_id} from cache_dir: {cfg.cache_dir}")
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": "auto", 
        "trust_remote_code": True,
        "cache_dir": cfg.cache_dir
    }
    
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)
    
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    logging.info(f"Applying LoRA to Qwen modules: {lora_target_modules}")
    lora_config = LoraConfig(
        r=cfg.lora_r, 
        lora_alpha=cfg.lora_alpha, 
        target_modules=lora_target_modules,
        bias="none", 
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base, lora_config)
    model.print_trainable_parameters()

    logging.info("Preparing dataset using STRICT Llama-adapted logic for Qwen...")
    train_ds = prepare_dataset(tok, cfg.keep_nei, cfg.cache_dir)
    
    if len(train_ds) == 0:
        logging.error("Training dataset is empty after preparation. Aborting.")
        raise ValueError("Training dataset is empty. Please check data preparation and filtering steps.")

    targs = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch,
        gradient_accumulation_steps=cfg.grad_acc,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=cfg.fp16 and not (cfg.bf16 if torch.cuda.is_bf16_supported() else False),
        bf16=cfg.bf16 if torch.cuda.is_bf16_supported() else False,
        logging_steps=cfg.logging_steps, 
        save_strategy="epoch",
        gradient_checkpointing=False, # Keep disabled
        logging_dir=f"{cfg.output_dir}/runs",
        report_to=["tensorboard"], 
    )

    logging.info(f"Using Training Arguments for Qwen (Strict Llama Logic Dataset): {targs}")

    return SFTTrainer(
        model=model, 
        tokenizer=tok, 
        train_dataset=train_ds,
        max_seq_length=cfg.max_seq_length, 
        dataset_text_field="text",  
        args=targs,
    )

# ------------------------- CLI (Qwen Specific Args) ----------------------------
def cli():
    ap = argparse.ArgumentParser(description="Fine-tune Qwen model for FEVER (Strict Llama-adapted data logic)")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct", help="Base Qwen model ID.")
    ap.add_argument("--token", default=None, help="Hugging Face token.")
    ap.add_argument("--output_dir", default="./finetuned_qwen_fever_strict_llama", 
                    help="Directory to save LoRA weights and logs.")
    ap.add_argument("--batch", type=int, default=16, help="Per device batch size.")
    ap.add_argument("--grad_acc", type=int, default=2, help="Gradient accumulation steps.")
    ap.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    # grad_ckpt is managed by TrainingArguments.gradient_checkpointing
    ap.add_argument("--fp16", type=bool, default=False, help="Enable FP16 training.")
    ap.add_argument("--bf16", type=bool, default=False, help="Enable BF16 training.")
    ap.add_argument("--keep_nei", action="store_true", help="Include NOT ENOUGH INFO.")
    ap.add_argument("--cache_dir", default=CACHE_DIR_TARGET, help="Cache directory.")
    ap.add_argument("--lora_r", type=int, default=8, help="LoRA r dimension.")
    ap.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    ap.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length.")
    ap.add_argument("--logging_steps", type=int, default=50, help="Log every N steps.")
    return ap.parse_args()

# ------------------------- MAIN ---------------------------
if __name__ == "__main__":
    try:
        import datasets
        import pyarrow
        import transformers 
        import torch 
        logging.info("--- Library Versions ---")
        logging.info(f"Python version: {os.sys.version.splitlines()[0]}")
        logging.info(f"Datasets version: {datasets.__version__}")
        logging.info(f"PyArrow version: {pyarrow.__version__}")
        logging.info(f"Transformers version: {transformers.__version__}")
        logging.info(f"Torch version: {torch.__version__}")
        logging.info("----------------------")
    except ImportError as e_import:
        logging.error(f"Could not import a library to check version: {e_import}")

    cfg = cli()
    logging.info(f"--- Qwen FEVER Fine-tuning (Strict Llama Logic Dataset) Configuration ---")
    for arg_name, value in vars(cfg).items():
        logging.info(f"  {arg_name}: {value}")
    logging.info(f"------------------------------------------\n")

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)
        logging.info(f"Created output directory: {cfg.output_dir}")

    if cfg.bf16 and not torch.cuda.is_bf16_supported():
        logging.warning("Warning: BF16 is requested but not supported. Disabling BF16.")
        cfg.bf16 = False
    if cfg.bf16 and cfg.fp16:
        logging.warning("Warning: Both BF16 and FP16 requested. Prioritizing BF16. Disabling FP16.")
        cfg.fp16 = False
    if not cfg.bf16 and not cfg.fp16 and torch.cuda.is_available():
        logging.warning("Warning: Neither BF16 nor FP16 enabled. Training in FP32.")

    trainer = build_trainer(cfg)
    
    logging.info("\nStarting fine-tuning training (Qwen model, Strict Llama-adapted data logic)...")
    try:
        trainer.train()
        logging.info("Training finished successfully.")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)
        # raise # Optionally re-raise

    logging.info(f"\nSaving LoRA adapter model to {cfg.output_dir}...")
    try:
        trainer.save_model(cfg.output_dir) 
        logging.info(f"LoRA adapter model saved to {cfg.output_dir}.")
    except Exception as e:
        logging.error(f"Could not save LoRA adapter model: {e}", exc_info=True)

    try:
        # Ensure tokenizer is saved. SFTTrainer might do this, but explicit is safer.
        if hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
            trainer.tokenizer.save_pretrained(cfg.output_dir)
            logging.info(f"Tokenizer saved to {cfg.output_dir}.")
        else:
            # Fallback if trainer.tokenizer is not available, try to save tokenizer from cfg
            # This path may not be hit if trainer setup is complete.
            tok_fallback = AutoTokenizer.from_pretrained(cfg.model_id, cache_dir=cfg.cache_dir, trust_remote_code=True)
            tok_fallback.save_pretrained(cfg.output_dir)
            logging.info(f"Fallback: Tokenizer saved to {cfg.output_dir}.")

    except Exception as e:
        logging.error(f"Could not save tokenizer: {e}", exc_info=True)

    logging.info("\nFine-tuning script (Qwen with Strict Llama-adapted data logic) completed.") 