import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from phaseA_llama1b_fever import WikiCache  # Assuming this is in PYTHONPATH or same dir
import argparse
import time
from tqdm import tqdm
import os

# --- CLI Argument Parsing ---
parser = argparse.ArgumentParser(description="Validate a Phase A model (LoRA or base).")
parser.add_argument(
    "--base_model_id",
    type=str,
    default="meta-llama/Llama-3.2-1B-Instruct",
    help="Base model ID from Hugging Face Hub.",
)
parser.add_argument(
    "--lora_path",
    type=str,
    default=None,
    help="Path to the LoRA weights directory. If not provided or --no_lora is used, only base model is validated.",
)
parser.add_argument(
    "--no_lora",
    action="store_true",
    help="If set, do not load LoRA weights and validate the base model directly.",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=1000,
    help="Number of samples from the dev set to validate on.",
)
parser.add_argument(
    "--batch_size", type=int, default=64, help="Batch size for prediction."
)
parser.add_argument(
    "--max_length", type=int, default=512, help="Max sequence length for tokenizer."
)
parser.add_argument(
    "--fixed_max_new_tokens",
    type=int,
    default=4,  # Adjusted default, Llama3 tokens might be shorter
    help="Fixed number of max_new_tokens for generation. If 0 or negative, dynamic calculation is used.",
)
parser.add_argument(
    "--token",
    type=str,
    default=None,
    help="Hugging Face token for private models (or use HF_TOKEN env var).",
)
parser.add_argument(
    "--load_in_4bit",
    action="store_true",
    help="Load model in 4-bit quantization to save memory.",
)
parser.add_argument(
    "--load_in_8bit",
    action="store_true",
    help="Load model in 8-bit quantization to save memory.",
)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=None,
    help="Directory to cache the model. If not provided, the default cache directory will be used.",
)
args = parser.parse_args()

BASE = args.base_model_id
LORA = args.lora_path if not args.no_lora and args.lora_path else None
NUM_SAMPLES_TO_VALIDATE = args.num_samples
BATCH_SIZE = args.batch_size
MAX_LENGTH = args.max_length
FIXED_MAX_NEW_TOKENS = args.fixed_max_new_tokens
HF_TOKEN = args.token
LOAD_IN_4BIT = args.load_in_4bit
LOAD_IN_8BIT = args.load_in_8bit
CACHE_DIR = args.cache_dir

# 检查GPU配置和优化设置
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(
        f"\n--- GPUs detected: {num_gpus} GPU(s) available! Using device_map='auto'. ---"
    )
    gpu = torch.cuda.get_device_properties(
        0
    )  # Get properties of the first GPU for info
    print(f"Primary GPU: {gpu.name} - {gpu.total_memory/1e9:.2f} GB")

    # 对高端GPU启用TF32
    if any(gpu_type in gpu.name for gpu_type in ["A100", "H100", "A6000", "RTX 4090"]):
        print("High-end GPU detected, enabling TF32 precision")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 自动确定量化设置，如果用户未明确指定 (based on primary GPU memory)
    if not LOAD_IN_4BIT and not LOAD_IN_8BIT:
        mem_gb = gpu.total_memory / 1e9
        LOAD_IN_4BIT = mem_gb < 24  # 对于小内存GPU自动使用4bit量化
        LOAD_IN_8BIT = 24 <= mem_gb < 40 and not LOAD_IN_4BIT  # 对于中等内存GPU使用8bit
        if LOAD_IN_4BIT or LOAD_IN_8BIT:
            print(
                f"Auto-determined quantization: 4bit={LOAD_IN_4BIT}, 8bit={LOAD_IN_8BIT}"
            )
else:
    print("\nNo GPUs detected. Model will run on CPU.")

print(f"--- Configuration ---")
print(f"Base Model ID: {BASE}")
if LORA:
    print(f"LoRA Path: {LORA}")
else:
    print(f"LoRA Path: Not loading LoRA (using base model only).")
print(
    f"Max New Tokens (Fixed): {FIXED_MAX_NEW_TOKENS if FIXED_MAX_NEW_TOKENS > 0 else 'Dynamically calculated'}"
)
print(f"Batch Size: {BATCH_SIZE}")
print(
    f"Quantization: {'4-bit' if LOAD_IN_4BIT else '8-bit' if LOAD_IN_8BIT else 'None (FP16/BF16 or FP32 on CPU)'}"
)
print(f"Cache Dir: {CACHE_DIR if CACHE_DIR else 'Default'}")
print(f"---------------------\n")


# Initialize Tokenizer
try:
    start_time = time.time()
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        BASE, use_fast=True, token=HF_TOKEN, cache_dir=CACHE_DIR  # 启用快速tokenizer
    )

    # 确保pad token存在
    if tok.pad_token is None:
        print("Tokenizer does not have a pad_token. Setting pad_token to eos_token.")
        tok.pad_token = tok.eos_token  # Common practice for Llama models
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"  # Important for batched generation
    print(f"Tokenizer loaded in {time.time() - start_time:.2f}s")

except Exception as e:
    print(f"Error loading tokenizer for base model {BASE}: {e}")
    exit()

print("\n--- Tokenizer Details (Llama 3 Focus) ---")
print(f"EOS token: '{tok.eos_token}', ID: {tok.eos_token_id}")
print(
    f"PAD token: '{tok.pad_token}', ID: {tok.pad_token_id}"
)  # Should be same as EOS for Llama3 if not set
print(f"BOS token: '{tok.bos_token}', ID: {tok.bos_token_id}")
print("--- End Tokenizer Details ---\n")


# 优化的模型加载代码 (using device_map='auto')
try:
    start_time = time.time()
    print("Loading model...")

    # 准备模型配置
    model_kwargs = {
        "token": HF_TOKEN,
        "cache_dir": CACHE_DIR,
        # torch_dtype will be bfloat16 on GPU, float32 on CPU
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": (
            "auto" if torch.cuda.is_available() else None
        ),  # Use "auto" for GPU, None for CPU (or let Transformers decide)
    }

    # 根据命令行参数和环境设置量化参数
    if LOAD_IN_4BIT:
        model_kwargs["load_in_4bit"] = True
        print("Using 4-bit quantization")
    elif LOAD_IN_8BIT:
        model_kwargs["load_in_8bit"] = True
        print("Using 8-bit quantization")

    model = AutoModelForCausalLM.from_pretrained(BASE, **model_kwargs)
    print(f"Successfully loaded base model '{BASE}'. Model is on: {model.device}")
    if LORA:
        print(f"Attempting to load LoRA weights from '{LORA}'...")
        # When using device_map="auto", PeftModel should also be loaded with device_map="auto" or compatible settings.
        # For simplicity, if the base model is already on multiple devices, applying LoRA might need care.
        # A common approach is to load LoRA onto the same device map or adapt.
        # However, PeftModel.from_pretrained typically handles this well with device_map="auto" on base.
        model = PeftModel.from_pretrained(model, LORA, device_map="auto")
        print(f"Successfully loaded LoRA weights from '{LORA}'.")

    model.eval()  # 设置为评估模式
    print(f"Model loaded in {time.time() - start_time:.2f}s")

except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 优化的数据集加载
try:
    start_time = time.time()
    print("Loading dataset...")
    dev = load_dataset("fever", "v1.0", split="labelled_dev")
    dev = dev.filter(lambda r: r["label"] != "NOT ENOUGH INFO")
    wiki = WikiCache()
    print(f"Dataset loaded in {time.time() - start_time:.2f}s")
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
    # 使用tokenizer的apply_chat_template方法（推荐）
    messages = [
        {"role": "system", "content": llama3_system_message},
        {
            "role": "user",
            "content": f"Evidence: {evid}\nClaim: {r['claim']}\nQuestion: Is this claim supported or refuted by the evidence?",
        },
    ]

    try:
        # 尝试使用官方chat_template
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )  # add_generation_prompt for assistant turn
        return prompt
    except Exception as e:
        # 如果失败，回退到手动构建格式
        print(f"Warning: Failed to apply chat template: {e}. Using manual format.")
        prompt_parts = [
            f"{tok.bos_token}" if tok.bos_token else "<|begin_of_text|>",
            f"<|start_header_id|>system<|end_header_id|>\n\n{llama3_system_message}<|eot_id|>",
            f"<|start_header_id|>user<|end_header_id|>\n\nEvidence: {evid}\nClaim: {r['claim']}\nQuestion: Is this claim supported or refuted by the evidence?<|eot_id|>",
            f"<|start_header_id|>assistant<|end_header_id|>\n\n",
        ]
        return "".join(prompt_parts)


dev_subset = dev.select(range(min(NUM_SAMPLES_TO_VALIDATE, len(dev))))
print(
    f"\n--- Running validation on the first {len(dev_subset)} samples of the dev set (Llama 3.2 Prompt Format). ---\n"
)

# 使用tqdm显示进度条
print("Preparing prompts...")
start_time = time.time()
prompts = []
for r in tqdm(dev_subset, desc="Building prompts"):
    prompts.append(to_llama3_prompt(r))
print(f"Prompts prepared in {time.time() - start_time:.2f}s")

gold = ["SUPPORTED" if l == "SUPPORTS" else "REFUTED" for l in dev_subset["label"]]


# 优化的batch_predict_generate函数
@torch.no_grad()
def batch_predict_generate(prompts_list, batch_s=BATCH_SIZE):
    preds = []
    inference_start_time = time.time()  # Renamed for clarity

    # 计算max_new_tokens
    if FIXED_MAX_NEW_TOKENS > 0:
        current_max_new_tokens = FIXED_MAX_NEW_TOKENS
    else:
        len_supported_tokens = len(tok.encode("SUPPORTED", add_special_tokens=False))
        len_refuted_tokens = len(tok.encode("REFUTED", add_special_tokens=False))
        current_max_new_tokens = max(len_supported_tokens, len_refuted_tokens, 1) + 2
        print(
            f"Using DYNAMICALLY calculated max_new_tokens for generation: {current_max_new_tokens}\n"
        )

    print(f"Using max_new_tokens for generation: {current_max_new_tokens}\n")

    # 使用tqdm显示进度条
    total_batches = (len(prompts_list) + batch_s - 1) // batch_s
    pbar = tqdm(total=total_batches, desc="Processing batches")

    # 处理示例打印逻辑
    print_example = True

    for i in range(0, len(prompts_list), batch_s):
        batch_processing_start_time = time.time()  # Renamed for clarity
        current_batch_prompts = prompts_list[i : i + batch_s]

        # 准备输入
        inputs = tok(
            current_batch_prompts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        # 获取模型的主要设备，用于移动输入张量
        # For device_map="auto", model.device might point to 'meta' or CPU if model is sharded.
        # It's safer to pick a target device, e.g., the device of the first parameter or cuda:0 if available.
        target_device = (
            next(model.parameters()).device
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        # 生成文本 - device_map="auto" handles internal distribution
        generate_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": current_max_new_tokens,
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id,
            "do_sample": False,
        }
        generated_ids = model.generate(**generate_kwargs)

        # 解码生成的文本
        generated_texts = []
        for k_idx in range(generated_ids.shape[0]):
            # The generated_ids might be on a different device than inputs if model is sharded.
            # Ensure prompt_length is calculated based on input_ids on its original device before slicing generated_ids.
            # However, tokenizer handles decoding from any device usually.
            prompt_length_tokens = inputs["input_ids"][k_idx].shape[0]
            # Slice generated_ids to get only the generated part
            # generated_ids already contains the prompt tokens if not handled by generate
            # For generate, it should only output new tokens if input_ids is passed.
            # Let's assume generated_ids contains prompt + generation for now, and we slice.
            # More robust: use generate(..., return_full_text=False) if available, or check HF docs.

            # Simpler slicing: if model.generate prepends input_ids to the output
            # This depends on the specific generate configuration and HF version.
            # Assuming generated_ids are on target_device or CPU after generate
            # If generate returns only new tokens, this slice would be just generated_ids[k_idx]
            full_output_ids = generated_ids[k_idx]
            generated_part_ids = full_output_ids[prompt_length_tokens:]

            decoded_text = tok.decode(
                generated_part_ids, skip_special_tokens=True
            ).strip()
            generated_texts.append(decoded_text)

        # 打印第一批的详细信息
        if print_example and i == 0:
            print("\n--- First Batch Prompts (first example) ---")
            if current_batch_prompts:
                print(f"Example 1 Prompt:\n{current_batch_prompts[0]}\n")
            print("--- End First Batch Prompts ---")
            print("\n--- First Batch Generated Text (first 5 examples) ---")
            for k_idx in range(min(5, len(generated_texts))):
                print(f"Generated for Ex {k_idx+1}: '{generated_texts[k_idx]}'")
            print("--- End First Batch Generated Text ---\n")
            print_example = False

        # 解析生成的文本
        for gen_text in generated_texts:
            normalized_gen_text = gen_text.upper().strip()
            first_word = normalized_gen_text.split()[0] if normalized_gen_text else ""

            if first_word.startswith("SUPPORTED"):
                preds.append("SUPPORTED")
            elif first_word.startswith("REFUTED"):
                preds.append("REFUTED")
            else:
                if normalized_gen_text.startswith("SUPPORTED"):  # Fallback
                    preds.append("SUPPORTED")
                elif normalized_gen_text.startswith("REFUTED"):  # Fallback
                    preds.append("REFUTED")
                else:
                    preds.append("OTHER")

        # 计算并显示批处理速度
        batch_processing_time = time.time() - batch_processing_start_time  # Renamed
        samples_per_sec = (
            len(current_batch_prompts) / batch_processing_time
            if batch_processing_time > 0
            else 0
        )
        pbar.set_postfix(
            samples_per_sec=f"{samples_per_sec:.1f}",
            batch_time=f"{batch_processing_time:.2f}s",
        )
        pbar.update(1)

    pbar.close()

    # 计算总处理速度
    total_inference_time = time.time() - inference_start_time  # Renamed
    avg_samples_per_sec = (
        len(prompts_list) / total_inference_time if total_inference_time > 0 else 0
    )
    print(
        f"Total processing time: {total_inference_time:.2f}s, Average speed: {avg_samples_per_sec:.1f} samples/sec"
    )

    return preds


# 验证过程
print("Starting validation...")
overall_start_time = time.time()  # Renamed
pred = batch_predict_generate(prompts)
validation_processing_time = time.time() - overall_start_time  # Renamed

# 计算准确率
correct_predictions = 0
other_predictions = 0
for p, g in zip(pred, gold):
    if p == g:
        correct_predictions += 1
    if p == "OTHER":
        other_predictions += 1

acc = correct_predictions / len(gold) if len(gold) > 0 else 0
total_processed = len(gold)

print(f"\n--- Validation Results ---")
print(
    f"Clean dev Accuracy: {acc * 100:.2f}%   ({correct_predictions}/{total_processed})"
)
if other_predictions > 0:
    print(
        f"Number of 'OTHER' predictions (neither SUPPORTED nor REFUTED detected): {other_predictions} out of {total_processed}"
    )
print(
    f"Total validation time (including prompt prep): {validation_processing_time:.2f}s, Average speed (overall): {total_processed/validation_processing_time:.1f} samples/sec"
    if validation_processing_time > 0
    else "N/A"
)
print("\nValidation finished.")
