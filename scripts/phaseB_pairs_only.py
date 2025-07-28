#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase B 训练（只用 700 对 adversarial 样本）
------------------------------------------------
• 输入：pairs_csv        —— 需含列 original_samples, adversarial_samples, agreed_labels
• 目标：让模型对"语义未变、标签翻转"的文本对保持一致预测（KL loss）
"""
import os, argparse, pandas as pd, torch, torch.nn.functional as F
from datasets import Dataset, Value
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments)
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
import pyarrow as pa # Import pyarrow for exception type

# --- 新增: 设置缓存目录环境变量 --- 
# Default cache directory for Hugging Face assets
# Hugging Face 资源的默认缓存目录
CACHE_DIR_TARGET = "/mnt/parscratch/users/acc22xl/huggingface_cache/" 
os.environ["HF_HOME"] = CACHE_DIR_TARGET
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR_TARGET, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR_TARGET, "models")
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_DIR_TARGET, "hub")
# --- 结束: 设置缓存目录环境变量 ---


# ========== 1. CSV → Dataset ==========
def load_adv_pairs(csv_path: str) -> Dataset:
    """读取包含样本对的 CSV，筛选 agreed_labels==0.0 的样本，
       并将其重构为包含 text, is_adv, pair_id 列的格式。
       每对原始/对抗样本会变成两条记录。
    """
    print(f"Loading and restructuring data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")

    # --- 检查必需列 ---
    required_cols = ['agreed_labels', 'original_samples', 'adversarial_samples']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV file '{csv_path}' is missing required columns: {missing_cols}")

    # --- 添加 pair_id (如果不存在) ---
    if 'pair_id' not in df.columns:
        print("Warning: 'pair_id' column not found in CSV. Generating pair IDs from index.")
        df['pair_id'] = df.index
    elif df['pair_id'].isnull().any():
         print("Warning: 'pair_id' column contains null values. Filling with index where null.")
         df['pair_id'] = df['pair_id'].fillna(pd.Series(df.index))
         
    # Ensure pair_id is integer type for proper sorting/grouping
    # 确保 pair_id 是整数类型以便正确排序/分组
    try:
         df['pair_id'] = df['pair_id'].astype(int)
    except (ValueError, TypeError):
         print("Warning: Could not convert 'pair_id' column to integer. Sorting might be affected.")


    # --- 使用 agreed_labels 进行筛选，并处理 0.0 ---
    original_count = len(df)

    def is_agreed_label_zero(label_value):
        if label_value is None: return False
        try: return float(label_value) == 0.0
        except (ValueError, TypeError): return False

    df_filtered = df[df['agreed_labels'].apply(is_agreed_label_zero)].copy() # Create a copy to avoid SettingWithCopyWarning
                                                                           # 创建副本以避免 SettingWithCopyWarning
    
    filtered_count = len(df_filtered)
    print(f"Filtered CSV based on 'agreed_labels == 0.0'. Original rows: {original_count}, Filtered pairs: {filtered_count}")
    if filtered_count == 0:
         print("Warning: No pairs remained after filtering for agreed_labels == 0.0. Check the input CSV.")
         # Return empty dataset if no pairs are left
         # 如果没有留下任何对，则返回空数据集
         return Dataset.from_dict({'text': [], 'is_adv': [], 'pair_id': [], 'labels': []})

    # --- 重构数据: 将样本对拆分为单独的行 ---
    original_part = df_filtered[['original_samples', 'pair_id']].copy()
    original_part.rename(columns={'original_samples': 'text'}, inplace=True)
    original_part['is_adv'] = 0

    adversarial_part = df_filtered[['adversarial_samples', 'pair_id']].copy()
    adversarial_part.rename(columns={'adversarial_samples': 'text'}, inplace=True)
    adversarial_part['is_adv'] = 1

    df_restructured = pd.concat([original_part, adversarial_part], ignore_index=True)

    # --- 按 pair_id 和 is_adv 排序以确保样本对相邻 ---
    df_restructured.sort_values(by=['pair_id', 'is_adv'], inplace=True)

    # --- 添加 labels 列 ---
    df_restructured["labels"] = -100

    print(f"Restructured data into {len(df_restructured)} individual samples (original and adversarial).")

    # Final check for required columns before creating Dataset
    # 创建数据集前最终检查必需的列
    final_expected_cols = ['text', 'is_adv', 'pair_id', 'labels']
    missing_final_cols = [col for col in final_expected_cols if col not in df_restructured.columns]
    if missing_final_cols:
         raise ValueError(f"Restructured DataFrame is missing expected columns: {missing_final_cols}")
         
    # Ensure 'text' column is string type
    # 确保 'text' 列是字符串类型
    df_restructured['text'] = df_restructured['text'].astype(str)
         
    # Select only necessary columns for the Dataset
    # 仅选择数据集所需的列
    df_final = df_restructured[final_expected_cols]

    return Dataset.from_pandas(df_final, preserve_index=False)


# ========== 2. Data collator ==========
def adv_collator(features):
    input_ids = [f["input_ids"].clone() for f in features]
    # --- 修改: 从全局变量获取 tokenizer ---
    # Assume 'tok' is accessible globally or passed differently if needed
    # 假设 'tok' 可全局访问，或者根据需要以不同方式传递
    global tok 
    if 'tok' not in globals() or tok is None:
         raise NameError("Tokenizer 'tok' is not defined globally for adv_collator.")
    # --- 结束修改 ---
    pad_id = tok.pad_token_id
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    attention_mask = (input_ids != pad_id).long()

    def stack(name):
        # Simplified check, assuming features list is not empty
        # 简化检查，假设 features 列表不为空
        if name not in features[0]:
            # Check if the key exists in *any* feature for better error message
            # 检查键是否存在于 *任何* 特征中以获得更好的错误消息
            key_exists = any(name in f for f in features)
            if not key_exists:
                 raise KeyError(f"Field '{name}' not found in any sample of the batch. Available keys in first sample: {list(features[0].keys())}")
            else:
                 # Handle cases where the key might be missing in some samples (though filter should prevent this for required keys)
                 # 处理键可能在某些样本中缺失的情况（尽管过滤器应该阻止这种情况发生对于必需的键）
                 print(f"[Warning] Field '{name}' missing in some batch samples, stacking available ones.")
                 values = [f[name] for f in features if name in f]
        else:
            values = [f[name] for f in features]
            
        if not values:
             raise ValueError(f"No values found for field '{name}' in the batch.")
             
        # Ensure all values are tensors if the first one is
        # 如果第一个值是张量，确保所有值都是张量
        if isinstance(values[0], torch.Tensor):
            return torch.stack(values)
        else:
            # Attempt conversion to tensor, handle potential type issues
            # 尝试转换为张量，处理潜在的类型问题
            try:
                return torch.tensor(values)
            except TypeError as e:
                print(f"Error converting field '{name}' to tensor. Values: {values}")
                raise TypeError(f"Could not convert values for '{name}' to tensor: {e}")


    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": stack("labels"),
        "pair_id": stack("pair_id"),
        # "semantic": stack("semantic"), # Removed semantic
        "is_adv": stack("is_adv"),
    }
    # print(f"Batch keys in collator: {list(batch_dict.keys())}") # Debug: 打印整理器中的批次键
    return batch_dict

# ========== 3. 自定义 Trainer ==========
class AdvTrainer(Trainer):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # print(">> DEBUG: entered compute_loss 进入！！！！") # Debug print
        # print(">> DEBUG: inside compute_loss; id=", id(self)) # Debug print

        # ----- 取出自定义字段 -----
        pair_id = inputs.pop("pair_id")
        # semantic = inputs.pop("semantic") # Removed semantic
        is_adv = inputs.pop("is_adv") # Keep is_adv if needed, e.g., for debugging

        # ----- 前向 -----
        # Ensure 'labels' is not passed to the model if it expects only input_ids/attention_mask
        # 如果模型只期望 input_ids/attention_mask，确保不将 'labels' 传递给模型
        model_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        outputs = model(**model_inputs)
        
        # Check if logits are directly available or nested
        # 检查 logits 是直接可用还是嵌套的
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            # Attempt to access logits if nested (e.g., in Seq2SeqLMOutput)
            # 尝试访问嵌套的 logits（例如，在 Seq2SeqLMOutput 中）
            try:
                logits = outputs[0] # Often the first element
            except (TypeError, IndexError):
                 raise AttributeError("Could not find 'logits' attribute or access logits as the first element in the model output.")

        # Assuming causal LM, take logits of the last token
        # 假设是因果 LM，取最后一个 token 的 logits
        # Ensure logits has 3 dimensions [batch, seq_len, vocab_size]
        # 确保 logits 有 3 个维度 [batch, seq_len, vocab_size]
        if len(logits.shape) != 3:
             raise ValueError(f"Expected model output logits to have 3 dimensions [batch, seq_len, vocab_size], but got shape {logits.shape}")
             
        last_token_logits = logits[:, -1, :]  # [B, vocab]

        # 1) CE：挂在 logits 上的 0 (No actual CE loss)
        # 1) CE：挂在 logits 上的 0 （没有实际的 CE 损失）
        loss_ce = last_token_logits.sum() * 0.0

        # 2) KL Loss Calculation
        # Assumes batch contains pairs: [orig1, adv1, orig2, adv2, ...]
        # 假设批次包含样本对：[orig1, adv1, orig2, adv2, ...]
        # Since filtering keeps only agreed_labels==0.0, all pairs are semantically consistent
        # 由于筛选只保留 agreed_labels==0.0，所有样本对都是语义一致的
        idx_o = torch.arange(0, last_token_logits.size(0), 2, device=last_token_logits.device)
        idx_a = idx_o + 1

        # Check if batch size is even and contains full pairs
        # 检查批次大小是否为偶数并包含完整的样本对
        if len(idx_a) > last_token_logits.size(0) // 2 or last_token_logits.size(0) % 2 != 0:
             print(f"Warning: Odd batch size ({last_token_logits.size(0)}) or mismatched pairs detected in KL loss calculation. Skipping KL for this batch.")
             loss_kl = last_token_logits.sum() * 0.0
        elif len(idx_o) == 0: # Handle empty batch case if filtering removed all
             loss_kl = last_token_logits.sum() * 0.0
        else:
            p = F.log_softmax(last_token_logits[idx_o], dim=-1) # Original samples log-probabilities
            q = F.softmax(last_token_logits[idx_a], dim=-1)     # Adversarial samples probabilities
            loss_kl = F.kl_div(p, q, reduction="batchmean", log_target=False) # Use log_target=False as q is softmax

        # Ensure loss_kl is a scalar
        # 确保 loss_kl 是一个标量
        if not isinstance(loss_kl, torch.Tensor) or loss_kl.numel() != 1:
             raise TypeError(f"Calculated KL loss is not a scalar tensor. Got: {loss_kl}")
             
        dummy = last_token_logits.sum() * 0.0  # Dummy tensor for graph connection if needed
                                               # 如果需要，用于图形连接的虚拟张量

        total_loss = dummy + loss_ce + self.alpha * loss_kl

        # Log the KL loss
        # 记录 KL 损失
        self.log({"kl_loss": loss_kl.item()})

        # Return format: (loss, outputs) if return_outputs=True, else loss
        # 返回格式：如果 return_outputs=True，则为 (loss, outputs)，否则为 loss
        # Note: Returning raw logits might be large; consider returning None or specific parts if needed
        # 注意：返回原始 logits 可能很大；如果需要，考虑返回 None 或特定部分
        return (total_loss, {'logits': last_token_logits}) if return_outputs else total_loss


# ========== 4. CLI ==========
def get_args():
    ap = argparse.ArgumentParser()
    # --- 修改: 移除 required=True ---
    ap.add_argument("--base_model", default="meta-llama/Llama-3.2-3B-Instruct",
                    help="基座模型或 HuggingFace 路径")
    ap.add_argument("--phaseA_dir",  default="phaseA_llama3B_L3prompt",
                    help="Phase A LoRA 权重目录（留空=不用）")
    ap.add_argument("--pairs_csv", default="./scripts_csv/train.csv",
                    help="包含样本对的 CSV 文件路径 (应包含 agreed_labels 列)") # 更新帮助信息
    ap.add_argument("--output_dir", default="./phaseB_new_llama3B_L3prompt_fromA_KLonly", # 更新默认输出目录名
                    help="保存 Phase B 权重的目录")
    # --- 结束修改 ---
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="KL loss 系数")
    ap.add_argument("--batch", type=int, default=8, 
                    help="批次大小 (应为偶数以进行样本对处理)") # 添加提示
    ap.add_argument("--grad_acc", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--cache_dir", default=CACHE_DIR_TARGET,
                    help="HuggingFace 模型/数据集缓存目录 (默认为环境变量设置)")
    ap.add_argument("--token", default="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",
                    help="HuggingFace token（如私有模型需登录）")
    args = ap.parse_args()
    
    # Add validation for batch size
    # 添加批次大小验证
    if args.batch % 2 != 0:
        print(f"Warning: Batch size ({args.batch}) is not even. KL divergence loss requires paired samples. Consider using an even batch size.")
        # Optionally raise error: raise ValueError("Batch size must be even for KL divergence pairing.")
        # 可选地引发错误：引发 ValueError("批次大小必须是偶数才能进行 KL 散度配对。")
        
    return args

# ========== 5. Main ==========
# --- 定义全局 tokenizer 变量 ---
# tok = None # 移除了顶层的 None 赋值
# --- 结束 ---

if __name__ == "__main__":
    cfg = get_args()

    # 5‑1. tokenizer
    # --- 修改: 将 tokenizer 赋值给全局变量 ---
    global tok # global 声明
    tok = None  # 在使用 global 声明后，但在实际赋值前初始化为 None
    tok = AutoTokenizer.from_pretrained(
        cfg.base_model, 
        use_fast=False, # AdvTrainer might need slow tokenizer features? Check if True works.
                        # AdvTrainer 可能需要慢分词器功能？检查 True 是否有效。
        cache_dir=cfg.cache_dir,
        token=cfg.token
    )
    # --- 结束修改 ---
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # 5‑2. Dataset：只用 pairs
    train_ds = load_adv_pairs(cfg.pairs_csv)
    
    # Check if dataset is empty after loading/filtering
    # 检查加载/筛选后数据集是否为空
    if len(train_ds) == 0:
        raise ValueError(f"Dataset loaded from '{cfg.pairs_csv}' is empty after filtering for agreed_labels == 0.0. Cannot proceed with training.")

    # --- 修改: 移除 'semantic' 的类型转换 ---
    # ★ cast dtype（int64）
    # Assuming 'is_adv' and 'pair_id' exist and need casting
    # 假设 'is_adv' 和 'pair_id' 存在并且需要转换类型
    columns_to_cast = []
    # Check existence rigorously before casting
    # 严格检查存在性后再进行转换
    if "is_adv" in train_ds.column_names and pd.api.types.is_numeric_dtype(train_ds['is_adv']): 
        columns_to_cast.append("is_adv")
    elif "is_adv" in train_ds.column_names:
         print(f"Warning: Column 'is_adv' found but is not numeric. Skipping casting to int64.")
         
    if "pair_id" in train_ds.column_names and pd.api.types.is_numeric_dtype(train_ds['pair_id']): 
        columns_to_cast.append("pair_id")
    elif "pair_id" in train_ds.column_names:
         print(f"Warning: Column 'pair_id' found but is not numeric. Skipping casting to int64.")

    for col in columns_to_cast:
        try:
            train_ds = train_ds.cast_column(col, Value("int64"))
        except (pa.lib.ArrowTypeError, ValueError, TypeError) as e:
            print(f"Warning: Could not cast column '{col}' to int64. Error: {e}. Check column contents in '{cfg.pairs_csv}'.")
            # Optionally raise error or handle specific cases
            # 可选地引发错误或处理特定情况
    # --- 结束修改 ---

    # 5‑3. Tokenisation
    def tok_fn(examples):
        # Assuming the input CSV has a 'text' column containing the actual text for original/adversarial samples
        # 假设输入 CSV 有一个 'text' 列，包含原始/对抗样本的实际文本
        if "text" not in examples:
             raise KeyError("The input data is missing the required 'text' column for tokenization.")
             
        tokens = tok(
            examples["text"],
            truncation=True,
            padding="max_length", # Consider 'do_not_pad' and handle padding in collator if max_length varies significantly
                                # 如果 max_length 差异很大，考虑 'do_not_pad' 并在整理器中处理填充
            max_length=512 # Make this configurable? e.g., args.max_seq_length
                           # 使其可配置？例如，args.max_seq_length
        )
        # Create dummy labels if needed by the model, typically -100 for CausalLM loss calculation
        # 如果模型需要，创建虚拟标签，对于 CausalLM 损失计算通常为 -100
        # tokens["labels"] = tokens["input_ids"].copy() # This might not be needed if loss is calculated externally like KL
                                                        # 如果像 KL 一样在外部计算损失，则可能不需要此操作
        tokens["labels"] = [-100] * len(tokens["input_ids"]) # Set all labels to -100 as CE loss is zeroed out
                                                            # 将所有标签设置为 -100，因为 CE 损失被置零

        # 保留其他字段
        # --- 修改: 移除 'semantic' ---
        keys_to_keep = []
        if "pair_id" in examples: keys_to_keep.append("pair_id")
        if "is_adv" in examples: keys_to_keep.append("is_adv")
        # --- 结束修改 ---
        
        for key in keys_to_keep:
            if key in examples:
                # Ensure values are lists for batch processing
                # 确保值是列表以便进行批处理
                if isinstance(examples[key], list):
                     tokens[key] = examples[key]
                else: # Handle potential non-list single values if map is not batched correctly
                     # 如果 map 没有正确批处理，处理潜在的非列表单个值
                     tokens[key] = [examples[key]] * len(examples['text']) # Repeat value for each text in batch
                                                                         # 对批次中的每个文本重复值

        # print(f"Keys in tokenized example: {list(tokens.keys())}") # Debug: 打印分词后示例中的键
        return tokens


    # Determine columns to remove carefully
    # 仔细确定要移除的列
    columns_to_remove = [col for col in train_ds.column_names if col not in ["text", "pair_id", "is_adv"]] # Keep text for tok_fn
                                                                                                         # 保留 text 以供 tok_fn 使用
    print(f"Columns to remove before map: {columns_to_remove}") # Debug: 打印映射前要移除的列

    train_ds = train_ds.map(
        tok_fn,
        batched=True,
        # remove_columns=["text"], # Remove text AFTER map if no longer needed
                                   # 如果不再需要，则在映射后移除 text
        load_from_cache_file=False # Disable caching for debugging or frequent changes
                                   # 禁用缓存以进行调试或频繁更改
    )

    # Define final columns needed for the collator and trainer
    # 定义整理器和训练器所需的最终列
    # --- 修改: 移除 'semantic' ---
    final_columns = ["input_ids", "attention_mask", "labels", "pair_id", "is_adv"]
    # --- 结束修改 ---
    
    # Remove columns not needed AFTER tokenization
    # 移除分词后不需要的列
    columns_to_remove_after_map = [col for col in train_ds.column_names if col not in final_columns]
    print(f"Columns to remove after map: {columns_to_remove_after_map}") # Debug: 打印映射后要移除的列
    if columns_to_remove_after_map:
       train_ds = train_ds.remove_columns(columns_to_remove_after_map)

    train_ds.set_format(
        type="torch",
        columns=final_columns
    )
    print(f"Final dataset columns: {train_ds.column_names}") # Debug: 打印最终数据集列

    # --- 修改: 更新必需字段检查，移除 'semantic' ---
    required_fields = ["input_ids", "attention_mask", "labels", "pair_id", "is_adv"]
    # --- 结束修改 ---
    # Filter samples missing required fields AFTER tokenization and formatting
    # 在分词和格式化后筛选缺少必需字段的样本
    original_len_before_filter = len(train_ds)
    train_ds = train_ds.filter(lambda ex: all(k in ex and ex[k] is not None for k in required_fields))
    if len(train_ds) < original_len_before_filter:
         print(f"Warning: Filtered out {original_len_before_filter - len(train_ds)} samples missing required fields after tokenization.")

    # Add a check here to ensure dataset is not empty before training
    # 在此处添加检查以确保数据集在训练前不为空
    if len(train_ds) == 0:
         raise ValueError("Dataset became empty after tokenization or final filtering. Cannot proceed.")
         
    # Sanity‑check
    # print(f"Sanity check: 'pair_id' in first sample? {'pair_id' in train_ds[0] if len(train_ds) > 0 else 'N/A (empty dataset)'}") # Debug
    # assert all("pair_id" in row for row in train_ds) # Might be slow for large datasets
                                                       # 对于大型数据集可能很慢

    # ========== 5‑4. Model (+LoRA)
    model_kwargs = {
        "device_map": "auto", 
        "torch_dtype": "auto", # Consider making this explicit based on fp16/bf16 args
                               # 考虑根据 fp16/bf16 参数明确指定
        "cache_dir": cfg.cache_dir,
        "token": cfg.token
    }
    # Handle precision explicitly
    # 明确处理精度
    if cfg.bf16 and torch.cuda.is_bf16_supported():
        model_kwargs["torch_dtype"] = torch.bfloat16
        print("Using bfloat16 precision for model.")
    elif cfg.fp16:
        model_kwargs["torch_dtype"] = torch.float16
        print("Using float16 precision for model.")
    else:
         print("Using default precision (likely float32) for model.")
         # Remove torch_dtype if defaulting to float32 to avoid potential conflicts
         # 如果默认为 float32，则移除 torch_dtype 以避免潜在冲突
         if model_kwargs.get("torch_dtype") == "auto":
              del model_kwargs["torch_dtype"]


    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, 
        **model_kwargs
    )

    # Load LoRA adapter if specified
    # 如果指定，加载 LoRA 适配器
    if cfg.phaseA_dir and os.path.exists(cfg.phaseA_dir):
        print(f"Loading LoRA adapter from {cfg.phaseA_dir}")
        model = PeftModel.from_pretrained(base_model, cfg.phaseA_dir)
        # Ensure adapter is trainable if further tuning is desired
        # 如果需要进一步调整，确保适配器是可训练的
        # model.set_adapter_trainable(adapter_name='default', mode=True) # Example if needed
        #                                                                # 如果需要，示例
    else:
        print("No Phase A LoRA directory specified or found. Using base model directly.")
        model = base_model # Use the base model without LoRA

    model.train() # Set model to training mode

    # ✅ 解冻全部参数（必要时 LoRA adapter 默认已解冻）
    # --- 修改: 恢复到仅训练 LoRA 适配器 (注释掉强制解冻) ---
    # print("Unfreezing all model parameters for full fine-tuning (as per previous working version).")
    # for param in model.parameters():
    #     param.requires_grad = True
    # --- 结束修改 ---

    # Verify trainable parameters AND explicitly set requires_grad
    # 验证可训练参数并显式设置 requires_grad
    trainable_params = 0
    all_param = 0
    print("Explicitly setting requires_grad for LoRA parameters...")
    for name, param in model.named_parameters():
        all_param += param.numel()
        # Check if 'lora_' is in the parameter name (standard PEFT naming)
        # 检查参数名称中是否包含 'lora_'（标准 PEFT 命名）
        if 'lora_' in name:
            param.requires_grad = True
            trainable_params += param.numel()
            # print(f"  Setting requires_grad=True for: {name}") # Optional: for debugging
        else:
            param.requires_grad = False
            # print(f"  Setting requires_grad=False for: {name}") # Optional: for debugging
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}"
    )
    # Check if any parameters were actually made trainable
    # 检查是否有任何参数实际变为可训练
    if trainable_params == 0:
        print("WARNING: No LoRA parameters found or set to trainable. Check adapter loading and PEFT setup.")
        # Optionally raise an error or proceed with caution
        # 可选地引发错误或谨慎进行
    # --- 结束修改 ---

    # 5‑5. Trainer
    trainer = AdvTrainer(
        alpha=cfg.alpha,
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        data_collator=adv_collator, # Use the custom collator
                                    # 使用自定义整理器
        args=TrainingArguments(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.batch,
            gradient_accumulation_steps=cfg.grad_acc,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.lr,
            warmup_ratio=0.05, # Consider making this configurable
                               # 考虑使其可配置
            fp16=cfg.fp16 and not cfg.bf16, # Use fp16 only if bf16 is not used
                                           # 仅当不使用 bf16 时才使用 fp16
            bf16=cfg.bf16 and torch.cuda.is_bf16_supported(),
            logging_dir=f"{cfg.output_dir}/logs",
            logging_strategy="steps",
            logging_steps=10, # Log more frequently? e.g., every step or 10 steps
                              # 更频繁地记录？例如，每步或每 10 步
            save_strategy="epoch",
            save_total_limit=1,
            dataloader_drop_last=True, # Important for KL loss pairing if last batch is odd
                                       # 如果最后一个批次是奇数，对于 KL 损失配对很重要
            remove_unused_columns=False, # Keep our custom columns ('pair_id', 'is_adv')
                                         # 保留我们的自定义列 ('pair_id', 'is_adv')
            # report_to="tensorboard" # Enable reporting to TensorBoard if needed
                                       # 如果需要，启用向 TensorBoard 报告
            # deepspeed=... # Add deepspeed config if used
                           # 如果使用，添加 deepspeed 配置
        )
    )

    # 5‑6. Train + save
    print("Starting training...")
    trainer.train()
    
    print("Training finished. Saving model...")
    # Save the final adapter (or full model if fully fine-tuned)
    # 保存最终的适配器（或完整模型，如果已完全微调）
    # PeftModel's save_pretrained saves only the adapter by default
    # PeftModel 的 save_pretrained 默认只保存适配器
    model.save_pretrained(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    print(f"✅ Model/Adapter and tokenizer saved to: {cfg.output_dir}")
    # Remove unnecessary variables at the end?
    # 最后移除不必要的变量？
    # save_strategy = "epoch" 
    # save_total_limit = 1
