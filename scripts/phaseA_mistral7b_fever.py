#!/usr/bin/env python3
# phaseA_mistral7b_fever.py —— 微调 Mistral‑7B‑Instruct‑v0.3 在 FEVER 二 / 三分类
# (采用 Mistral v0.3 对话格式)

import os, argparse
from typing import Dict
from datasets import load_dataset, Dataset
from huggingface_hub import login as hf_login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

CACHE_DIR_TARGET = "/mnt/parscratch/users/acc22xl/huggingface"
os.environ["HF_HOME"] = CACHE_DIR_TARGET
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR_TARGET
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR_TARGET
os.environ["HF_METRICS_CACHE"] = CACHE_DIR_TARGET

# ------------------------- Wiki 缓存 -------------------------
class WikiCache:
    """
    与原脚本保持一致：整页缓存 + 行号到句子映射
    """
    def __init__(self, cache_dir=None):
        wiki = load_dataset("fever", "wiki_pages",
                            cache_dir=cache_dir, trust_remote_code=True)["wikipedia_pages"]
        self._idx   = {row["id"]: i for i, row in enumerate(wiki)}
        self._data  = wiki
        self._cache: Dict[str, Dict[int, str]] = {}

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

# ------------------------- construct prompts -------------------------
# Mistral‑7B‑Instruct v0.3 uses <|im_start|> / <|im_end|> markers
mistral_system_message_train = (
    "You are a fact‑checking assistant. "
    "Given EVIDENCE and a CLAIM, reply with SUPPORTED or REFUTED. "
    "Do not output anything else."
)
mistral_system_message_train_nei = (
    "You are a fact‑checking assistant. "
    "Given EVIDENCE and a CLAIM, reply with SUPPORTED, REFUTED, or NOT ENOUGH INFO. "
    "Do not output anything else."
)

def build_prompt_mistral(sys_msg_content: str, evidence: str, claim: str) -> str:
    """
    Returns a prompt formatted for Mistral Instruct v0.3 chat format.
    The assistant's response is not yet answered, so it can be appended directly.
    """
    prompt_parts = [
        "<s>",
        f"<|im_start|>system\n{sys_msg_content}<|im_end|>\n",
        f"<|im_start|>user\nEvidence: {evidence}\nClaim: {claim}\n"
        "Question: Is this claim supported or refuted by the evidence?<|im_end|>\n",
        "<|im_start|>assistant\n"  
    ]
    return "".join(prompt_parts)

# ---------------------- 数据集准备 --------------------------
def prepare_dataset(tokenizer: AutoTokenizer, keep_nei=False, cache_dir=None) -> Dataset:
    """
    与原版保持同一逻辑，但调用 Mistral prompt 构造函数。
    关键点：将动态获取的 tokenizer.eos_token 追加到文本末尾，
    并确认所有样本都含有 EOS，避免 SFTTrainer 自动再加一次。
    """
    raw  = load_dataset("fever", "v1.0", split="train",
                        cache_dir=cache_dir, trust_remote_code=True)
    print(f"DEBUG phaseA: Loaded raw train set, length = {len(raw)}")
    wiki = WikiCache(cache_dir)

    current_sys_msg = (mistral_system_message_train_nei
                       if keep_nei else mistral_system_message_train)

    eos_token_to_use = tokenizer.eos_token
    if eos_token_to_use is None:          # 不太可能发生，但仍做防护
        print("Warning: tokenizer.eos_token is None, setting to </s> manually.")
        eos_token_to_use = "</s>"

    print(f"prepare_dataset will use EOS token: '{eos_token_to_use}'")

    def convert(ex):
        lab = ex["label"]
        if lab == "NOT ENOUGH INFO" and not keep_nei:
            return None

        if lab == "NOT ENOUGH INFO":
            ev, y_text = "No evidence provided.", "NOT ENOUGH INFO"
        elif lab == "SUPPORTS":
            ev, y_text = wiki.sent(ex["evidence_id"], ex["evidence_sentence_id"]), "SUPPORTED"
        elif lab == "REFUTES":
            ev, y_text = wiki.sent(ex["evidence_id"], ex["evidence_sentence_id"]), "REFUTED"
        else:
            return None

        prompt_part = build_prompt_mistral(current_sys_msg, ev, ex["claim"])
        full_text   = prompt_part + y_text + eos_token_to_use
        return {"text": full_text}

    ds = raw.map(convert, remove_columns=raw.column_names)
    return ds.filter(lambda x: x is not None)

# ------------------------- Trainer --------------------------
def build_trainer(cfg) -> SFTTrainer:
    # --- 登录（可选） ---
    if cfg.token or os.getenv("HF_TOKEN"):
        hf_login(token=cfg.token or os.getenv("HF_TOKEN"))

    # --- Tokenizer ---
    tok = AutoTokenizer.from_pretrained(
        cfg.model_id,
        add_eos_token=False,   # 文本已手动附加 EOS
        use_fast=False
    )
    if tok.pad_token is None:
        print("Tokenizer 无 pad_token，使用 eos_token 作为 pad_token.")
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    print("\n--- Tokenizer Details ---")
    print(f"EOS token: '{tok.eos_token}' (ID={tok.eos_token_id})")
    print(f"PAD token: '{tok.pad_token}' (ID={tok.pad_token_id}), side={tok.padding_side}")
    print("--------------------------\n")

    # --- Base model ---
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        device_map="auto",
        torch_dtype="auto"
    )

    # --- LoRA ---
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        lora_dropout=0.05
    )
    model = get_peft_model(base, lora_config)
    model.print_trainable_parameters()

    # --- Dataset ---
    train_ds = prepare_dataset(tok, cfg.keep_nei, cfg.cache_dir)

    # --- TrainingArgs ---
    targs = TrainingArguments(
        output_dir           = cfg.output_dir,
        per_device_train_batch_size = cfg.batch,
        gradient_accumulation_steps = cfg.grad_acc,
        num_train_epochs     = cfg.epochs,
        learning_rate        = cfg.lr,
        warmup_ratio         = 0.05,
        lr_scheduler_type    = "cosine",
        fp16                 = cfg.fp16,
        bf16                 = cfg.bf16,
        logging_steps        = 100,
        save_strategy        = "epoch",
        gradient_checkpointing = cfg.grad_ckpt,
        logging_dir          = f"{cfg.output_dir}/runs",
    )

    return SFTTrainer(
        model            = model,
        tokenizer        = tok,
        train_dataset    = train_ds,
        max_seq_length   = 512,
        dataset_text_field= "text",
        args             = targs
    )

# ------------------------- CLI ----------------------------
def cli():
    ap = argparse.ArgumentParser(
        description="Fine‑tune Mistral‑7B‑Instruct‑v0.3 on FEVER fact‑checking")
    ap.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.3",
                    help="Base model ID.")
    ap.add_argument("--token", default = "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",
                    help="Hugging Face token (if required).")
    ap.add_argument("--output_dir", default="./phaseA_mistral7B_fever",
                    help="Directory to save LoRA weights and logs.")
    ap.add_argument("--batch", type=int, default=16,
                    help="Per device batch size (7B 模型通常显存更占).")
    ap.add_argument("--grad_acc", type=int, default=2,
                    help="Gradient accumulation steps.")
    ap.add_argument("--epochs", type=int, default=1,
                    help="Number of training epochs.")
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="Learning rate.")
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="Enable gradient checkpointing.")
    ap.add_argument("--fp16", action="store_true",
                    help="Enable FP16 training.")
    ap.add_argument("--bf16", action="store_true",
                    help="Enable BF16 training.")
    ap.add_argument("--keep_nei", action="store_true",
                    help="Include NOT ENOUGH INFO for 3‑way classification.")
    ap.add_argument("--cache_dir", default=CACHE_DIR_TARGET, help="Cache directory.")
    return ap.parse_args()

# ------------------------- MAIN ---------------------------
if __name__ == "__main__":
    cfg = cli()
    print("--- Training Configuration ---")
    for arg, val in vars(cfg).items():
        print(f"{arg}: {val}")
    print("-------------------------------\n")

    os.makedirs(cfg.output_dir, exist_ok=True)

    trainer = build_trainer(cfg)

    print("\nStarting training...")
    trainer.train()
    print("Training finished.")

    print(f"\nSaving model to {cfg.output_dir}...")
    trainer.save_model(cfg.output_dir)
    print("Model saved.")