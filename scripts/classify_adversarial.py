# scripts/classify_adversarial.py

import pandas as pd
import torch
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from tqdm import tqdm
from peft import PeftModel

# Correct the target cache directory path
CACHE_DIR_TARGET = "/mnt/parscratch/users/acc22xl/huggingface_cache/"
os.environ["HF_HOME"] = CACHE_DIR_TARGET
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR_TARGET
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR_TARGET
os.environ["HF_METRICS_CACHE"] = (
    CACHE_DIR_TARGET  # Though less common, update for consistency
)


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def construct_prompts_for_llama3(df, tokenizer):
    original_prompt_messages = []
    adversarial_prompt_messages = []
    skipped_samples = []  # record the indices of samples that are skipped

    # Llama 3 system message
    system_message_content = (
        "You are a fact-checking assistant. "
        "Given EVIDENCE and a CLAIM, reply with exactly one token: SUPPORTED or REFUTED. "
        "Do not output anything else."
    )

    for index, row in df.iterrows():
        original_sample = row["original_samples"]
        adversarial_sample = row["adversarial_samples"]

        if not isinstance(original_sample, str):
            print(f"Skipping sample at index {index}: Original sample not a string.")
            skipped_samples.append(index)
            continue
        original_sample = original_sample.strip()

        if not isinstance(adversarial_sample, str):
            print(f"Skipping sample at index {index}: Adversarial sample not a string.")
            skipped_samples.append(index)
            continue
        adversarial_sample = adversarial_sample.strip()

        if "~" not in original_sample or "~" not in adversarial_sample:
            print(
                f"Skipping sample at index {index}: Missing '~' separator in one or both samples."
            )
            skipped_samples.append(index)
            continue

        original_parts = original_sample.split("~", 1)
        if (
            len(original_parts) != 2
            or not original_parts[0].strip()
            or not original_parts[1].strip()
        ):
            print(
                f"Skipping sample at index {index}: Improperly formatted original sample."
            )
            skipped_samples.append(index)
            continue

        adversarial_parts = adversarial_sample.split("~", 1)
        if (
            len(adversarial_parts) != 2
            or not adversarial_parts[0].strip()
            or not adversarial_parts[1].strip()
        ):
            print(
                f"Skipping sample at index {index}: Improperly formatted adversarial sample."
            )
            skipped_samples.append(index)
            continue

        evidence_original, claim_original = original_parts
        evidence_adversarial, claim_adversarial = adversarial_parts

        # Construct user message content
        user_message_original_content = (
            f"Evidence: {evidence_original.strip()}\\n"
            f"Claim: {claim_original.strip()}\\n"
            "Question: Is the claim supported or refuted by the evidence?\\n"
            "Answer:"
        )

        user_message_adversarial_content = (
            f"Evidence: {evidence_adversarial.strip()}\\n"
            f"Claim: {claim_adversarial.strip()}\\n"
            "Question: Is the claim supported or refuted by the evidence?\\n"
            "Answer:"
        )

        # Llama 3 chat format
        original_messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_original_content},
            # The model should generate the assistant's response.
            # We add a placeholder for the assistant role here to guide the model.
            # The tokenizer.apply_chat_template will typically add the assistant prompt.
        ]

        adversarial_messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_adversarial_content},
        ]

        original_prompt_messages.append(original_messages)
        adversarial_prompt_messages.append(adversarial_messages)

    total_samples = len(df)
    valid_samples = total_samples - len(skipped_samples)

    print(f"Total samples: {total_samples}")
    print(f"Skipped samples: {len(skipped_samples)}")
    print(f"Valid samples: {valid_samples}")

    return (
        original_prompt_messages,
        adversarial_prompt_messages,
        skipped_samples,
        valid_samples,
    )


def load_model(model_name, token=None, lora_path=None, cache_dir=None):
    """加载模型和分词器，支持多种加载选项，包括LoRA"""
    print("Loading model...")

    # 检查GPU可用性
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name} - {gpu.total_memory/1e9:.2f} GB")

        # 对高端GPU启用TF32
        if any(
            gpu_type in gpu.name for gpu_type in ["A100", "H100", "A6000", "RTX 4090"]
        ):
            print("High-end GPU detected, enabling TF32 precision")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # 根据GPU内存确定量化参数
        mem_gb = gpu.total_memory / 1e9
        use_4bit = mem_gb < 24  # 小内存GPU使用4bit量化
        use_8bit = 24 <= mem_gb < 40 and not use_4bit  # 中等内存使用8bit
        print(f"Memory optimization: 4bit={use_4bit}, 8bit={use_8bit}")
    else:
        print("GPU not available, using CPU")
        use_4bit = False
        use_8bit = False

    # 准备模型配置
    model_kwargs = {
        "use_auth_token": token,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "cache_dir": cache_dir,
    }

    # 加载tokenizer
    print(f"Loading tokenizer for base model: {model_name} using cache: {cache_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_auth_token=token, cache_dir=cache_dir
    )

    # 重要：Llama 3 typically expects left padding for batched generation.
    tokenizer.padding_side = "left"

    # 确保pad token存在
    if tokenizer.pad_token is None:
        # For Llama 3, if a pad token is not explicitly set,
        # it's often set to eos_token. However, it's better to ensure it's explicitly set.
        # Check if model config specifies a pad_token_id, otherwise use eos_token_id.
        # If a specific pad_token is needed, it should be added to the tokenizer vocab
        # and resized model embeddings. For now, eos_token is a common choice.
        print("pad_token is None. Setting pad_token = eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        # Make sure the model's pad_token_id is also updated if necessary
        # model.config.pad_token_id = tokenizer.pad_token_id

    # 加载基础模型
    try:
        print(f"Loading base model: {model_name} with config: {model_kwargs}")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as e:
        print(f"Error loading base model: {e}")
        print("Falling back to basic loading for base model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=token,
            torch_dtype=torch.float32,
            cache_dir=cache_dir,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")

    # 如果提供了LoRA路径，则加载并合并LoRA权重
    if lora_path:
        print(f"LoRA path provided: {lora_path}")
        if not os.path.exists(lora_path):
            print(
                f"Warning: LoRA path {lora_path} does not exist. Using base model only."
            )
        else:
            try:
                print(f"Loading and merging LoRA weights from {lora_path}...")
                model = PeftModel.from_pretrained(model, lora_path)
                model = model.merge_and_unload()
                print("LoRA weights merged successfully.")
            except Exception as e:
                print(
                    f"Error loading or merging LoRA weights: {e}. Using base model only."
                )

    # 设置评估模式
    model.eval()
    device = next(model.parameters()).device
    print(f"Model (final) loaded successfully on {device}")

    # It's good practice to ensure the model's config also reflects the pad_token_id used by the tokenizer
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        print(
            f"Setting model.config.pad_token_id to tokenizer.pad_token_id ({tokenizer.pad_token_id})"
        )
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


def classify_samples(tokenizer, model, prompt_messages_list, batch_size=8):
    predictions = []
    device = next(model.parameters()).device

    # 优化GPU批处理大小
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        mem_gb = gpu.total_memory / 1e9
        # 根据GPU内存动态调整批大小
        suggested_batch = max(
            1, min(int(mem_gb // 5), 32)
        )  # 每5GB内存约一个批次，上限32
        if batch_size == 1:  # 用户未明确设置批大小
            batch_size = suggested_batch
            print(
                f"Auto-tuned batch size: {batch_size} based on {mem_gb:.1f}GB GPU memory"
            )
        else:
            print(
                f"Using user-specified batch size: {batch_size} (GPU: {mem_gb:.1f}GB)"
            )
    else:
        print(f"Using CPU with batch size: {batch_size}")

    # 显示进度条
    total_batches = (len(prompt_messages_list) + batch_size - 1) // batch_size
    pbar = tqdm(total=total_batches, desc="Processing batches")

    start_time = time.time()
    # 批量处理
    for i in range(0, len(prompt_messages_list), batch_size):
        batch_messages = prompt_messages_list[i : i + batch_size]
        batch_start = time.time()

        # Apply Llama 3 chat template
        # add_generation_prompt=True adds the prompt for the assistant's turn
        batch_prompts_formatted = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in batch_messages
        ]

        try:
            # 使用padding处理批次 - 确保左侧填充
            inputs = tokenizer(
                batch_prompts_formatted, padding=True, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,  # 只生成少量token
                    do_sample=False,
                    temperature=None,  # 移除不必要的温度设置
                    top_p=None,  # 移除不必要的top_p设置
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # 处理每个输出
            for j, output in enumerate(outputs):
                # 获取输入长度以找到生成的部分
                input_length = inputs.input_ids.shape[1]  # 修正输入长度计算
                generated_tokens = output[input_length:]

                # 解码生成的文本
                generated_text = tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                ).strip()

                # 解析答案
                prediction = parse_answer(generated_text)
                predictions.append(prediction)

                # 打印首批样本的详细信息
                if i == 0 and j < 3:
                    print(f"  Sample {j+1}:")
                    print(f"    Prompt: {batch_prompts_formatted[j]}")
                    print(f"    Generated: {generated_text}")
                    print(f"    Prediction: {prediction}")

            # 计算并显示批处理速度
            batch_time = time.time() - batch_start
            samples_per_sec = len(batch_prompts_formatted) / batch_time
            pbar.set_postfix(
                samples_per_sec=f"{samples_per_sec:.1f}",
                batch_time=f"{batch_time:.2f}s",
            )

        except Exception as e:
            print(f"Error in batch processing: {e}")
            # 单条回退处理
            for j, prompt in enumerate(batch_prompts_formatted):
                try:
                    print(f"Processing individual prompt {i+j+1}")
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        output = model.generate(
                            **inputs,
                            max_new_tokens=10,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                        )

                    # 解码生成的文本
                    generated_text = tokenizer.decode(
                        output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
                    ).strip()

                    # 解析答案
                    prediction = parse_answer(generated_text)
                    predictions.append(prediction)

                except Exception as inner_e:
                    print(f"Error in single sample processing: {inner_e}")
                    predictions.append("UNKNOWN")

        pbar.update(1)

    pbar.close()

    # 计算总处理速度
    total_time = time.time() - start_time
    avg_samples_per_sec = len(prompt_messages_list) / total_time
    print(
        f"Total processing time: {total_time:.2f}s, Average speed: {avg_samples_per_sec:.1f} samples/sec"
    )

    # 打印预测分布
    supported_count = predictions.count("SUPPORTED")
    refuted_count = predictions.count("REFUTED")
    unknown_count = predictions.count("UNKNOWN")

    print(f"\nPrediction distribution:")
    print(
        f"  SUPPORTED: {supported_count} ({supported_count/len(predictions)*100:.1f}%)"
    )
    print(f"  REFUTED: {refuted_count} ({refuted_count/len(predictions)*100:.1f}%)")
    if unknown_count > 0:
        print(f"  UNKNOWN: {unknown_count} ({unknown_count/len(predictions)*100:.1f}%)")

    return predictions


def parse_answer(output_text):
    # 尝试从模型输出中提取 "SUPPORTED" 或 "REFUTED"
    # Llama 3 might add its own conversational fluff or structure, so be robust.

    # First, look for the exact keywords in uppercase
    if "SUPPORTED" in output_text:
        return "SUPPORTED"
    elif "REFUTED" in output_text:
        return "REFUTED"

    # Fallback: try lowercase and then try to infer from common phrasing
    # This part needs to be carefully tuned based on observed model outputs.
    # For Llama 3, it should be quite good at following the "exactly one token" instruction
    # if the prompt is clear.

    cleaned_output = output_text.strip().upper()  # Normalize to uppercase

    # Check again after normalizing
    if "SUPPORTED" in cleaned_output:
        return "SUPPORTED"
    elif "REFUTED" in cleaned_output:
        return "REFUTED"

    # If still not found, it's unrecognized
    print(
        f"Unrecognized answer: '{output_text}' (defaulting to REFUTED for safety, but please check)"
    )
    return "REFUTED"  # Defaulting to REFUTED might not always be correct. Consider how to handle.


def compare_results(
    df,
    original_predictions,
    adversarial_predictions,
    valid_samples,
    export_flipped_csv=None,
):
    # Add prediction result columns
    # Ensure indices align correctly, especially if some samples were skipped.
    # df is the original dataframe, original_predictions/adversarial_predictions are lists for valid samples.

    # Create series with NaNs for all original indices
    original_pred_series = pd.Series([None] * len(df), index=df.index)
    adversarial_pred_series = pd.Series([None] * len(df), index=df.index)

    # Get valid indices (those not skipped)
    valid_indices = df.index.drop(
        df[df["original_samples"].isna()].index
    )  # Assuming NaNs mark skipped or initially bad rows
    # If skipped_samples list was maintained, use that to get valid_indices more directly:
    # all_indices = df.index.tolist()
    # valid_indices = [idx for idx in all_indices if idx not in skipped_samples_from_construct_prompts]
    # For now, relying on the structure that predictions align with valid samples.

    # This assumes original_predictions and adversarial_predictions only contain results for valid, processed samples.
    # We need to map these back to the original DataFrame's indices.

    # Let's refine this: construct_prompts returns 'skipped_samples' which are indices from the input df.
    # We can use this to correctly align.

    # Get all indices from the DataFrame
    all_df_indices = df.index.tolist()

    # Determine valid indices by excluding skipped ones
    # This assumes `skipped_samples` contains the original DataFrame indices that were skipped
    # This part will need the `skipped_samples` list from the prompt construction step to be passed here
    # For now, we'll assume a direct mapping for simplicity if `skipped_samples` is not available here.
    # A better approach is to pass `skipped_samples` to this function or handle it in main.

    # Placeholder: This alignment needs to be robust.
    # If `valid_samples` corresponds to the length of `original_predictions`,
    # and we know which samples were *not* skipped, we can map them.

    # Let's assume `original_predictions` and `adversarial_predictions` are for the samples
    # that were *not* skipped during prompt construction.
    # The `df` here is the full dataframe. We need to put predictions into the correct rows.

    # Create temporary columns filled with a placeholder (e.g., np.nan or an empty string)
    df["original_prediction"] = pd.NA
    df["adversarial_prediction"] = pd.NA

    # Iterate through the original DataFrame and fill predictions if the sample was processed
    # This requires knowing which original indices correspond to the predictions.
    # If `construct_prompts_for_llama3` returns `skipped_samples` (indices of df),
    # then we can create a list of processed_indices.

    # Simplification: Assuming predictions directly map to the first `valid_samples` rows of `df`
    # that were not skipped. This is often the case if `df` is filtered *before* this function.
    # However, the current `load_data` loads the whole df.

    # Robust approach:
    # 1. `construct_prompts_for_llama3` returns `skipped_samples_indices` (original df indices)
    # 2. In `main`, filter `df` to get `processed_df` before calling `classify_samples` OR
    # 3. In `main`, pass `skipped_samples_indices` to `compare_results`.

    # For this edit, I'll assume `original_predictions` and `adversarial_predictions`
    # are lists that correspond to the rows of `df` that were NOT skipped.
    # The `main` function will need to handle the alignment.
    # This function will just add columns based on the provided lists.

    pred_idx_orig = 0
    pred_idx_adv = 0

    # We need a way to know which rows in df were processed.
    # Let's assume for now that the `main` function will handle providing df
    # that only contains processed rows, or that `original_predictions` has NaNs for skipped.
    # The provided `df` here is the full one.

    # A simple way if the lists are dense for processed samples:
    processed_indices = [
        idx
        for idx in df.index
        if idx not in df.attrs.get("skipped_samples_indices", [])
    ]

    if len(processed_indices) == len(original_predictions) and len(
        processed_indices
    ) == len(adversarial_predictions):
        df.loc[processed_indices, "original_prediction"] = original_predictions
        df.loc[processed_indices, "adversarial_prediction"] = adversarial_predictions
    else:
        print(
            "Warning: Mismatch in length between processed indices and predictions. Predictions may not be aligned correctly."
        )
        # Fallback to direct assignment if lengths match df (less robust if there were skips)
        if len(df) == len(original_predictions):
            df["original_prediction"] = original_predictions
        if len(df) == len(adversarial_predictions):
            df["adversarial_prediction"] = adversarial_predictions

    df["comparison_result"] = "Not Run"  # Default

    # --- New: Handle 'correctness' column and calculate Clean Accuracy ---
    if "correctness" not in df.columns:
        print(
            "\nWarning: 'correctness' column not found. Skipping Clean Accuracy and related metrics."
        )
        clean_accuracy = 0
        df["original_is_correct"] = False  # Placeholder
        num_correct_original_samples_valid_gt = 0
    else:
        # Map 'correctness' values (e.g., "SUPPORTS" -> "SUPPORTED")
        df["mapped_correctness"] = (
            df["correctness"]
            .str.upper()
            .replace(
                {
                    "SUPPORTS": "SUPPORTED",
                    "REFUTES": "REFUTED",
                    # Add other potential mappings here if needed
                }
            )
        )
        # Consider only valid mapped labels for accuracy calculations
        valid_ground_truth_labels = ["SUPPORTED", "REFUTED"]
        df["is_valid_ground_truth"] = df["mapped_correctness"].isin(
            valid_ground_truth_labels
        )

        num_samples_with_valid_gt = df["is_valid_ground_truth"].sum()

        df["original_is_correct"] = (
            df["original_prediction"] == df["mapped_correctness"]
        ) & df["is_valid_ground_truth"]
        num_correct_original_samples_valid_gt = df["original_is_correct"].sum()

        if num_samples_with_valid_gt > 0:
            clean_accuracy = (
                num_correct_original_samples_valid_gt / num_samples_with_valid_gt
            )
        else:
            clean_accuracy = 0
        print(
            f"\nClean Accuracy (on original samples with valid ground truth): {clean_accuracy:.2%} ({num_correct_original_samples_valid_gt}/{num_samples_with_valid_gt})"
        )

    # Detect if classification result flipped
    df["prediction_flipped"] = df["original_prediction"] != df["adversarial_prediction"]

    # Calculate standard Flip Rate (based on all valid_samples)
    # valid_samples is the count of samples for which prompts were successfully built
    flipped_samples_count = df["prediction_flipped"].sum()
    overall_flip_rate = (
        flipped_samples_count / valid_samples if valid_samples > 0 else 0
    )

    # --- Calculate metrics for meaning-preserving samples ---
    meaning_preserving_df = df[df["agreed_labels"] == 0].copy()
    num_meaning_preserving = len(meaning_preserving_df)
    flips_in_meaning_preserving = meaning_preserving_df["prediction_flipped"].sum()

    # Original "Similarity-Weighted Flip Rate" (flips in meaning_preserving / total valid_samples)
    # Renaming for clarity based on its calculation
    flip_rate_mp_vs_all = (
        flips_in_meaning_preserving / valid_samples if valid_samples > 0 else 0
    )

    # --- New: Calculate metrics for (Meaning-Preserving AND Originally Correctly Classified) samples ---
    if "correctness" in df.columns:
        # Filter for meaning-preserving samples that were also originally correct and had valid ground truth
        correct_and_meaning_preserving_df = meaning_preserving_df[
            meaning_preserving_df["original_is_correct"]
        ]
        num_correct_and_mp = len(correct_and_meaning_preserving_df)

        flips_in_correct_and_mp = correct_and_meaning_preserving_df[
            "prediction_flipped"
        ].sum()

        robust_flip_rate = (
            flips_in_correct_and_mp / num_correct_and_mp
            if num_correct_and_mp > 0
            else 0
        )

        print(
            f"Meaning-Preserving & Correctly Classified Original Samples: {num_correct_and_mp}"
        )
        print(
            f"Flip Rate (for Meaning-Preserving & Correctly Classified Originals): {robust_flip_rate:.2%} ({flips_in_correct_and_mp}/{num_correct_and_mp})"
        )
        # 新增：Meaning-Preserving Flips / Meaning-Preserving & Correctly Classified Original Samples
        if num_correct_and_mp > 0:
            mp_flip_rate_targeted = flips_in_meaning_preserving / num_correct_and_mp
        else:
            mp_flip_rate_targeted = float("nan")
        print(
            f"Targeted MP Flip Rate (Meaning-Preserving Flips / MP & Correctly Classified Originals): {mp_flip_rate_targeted:.2%} ({flips_in_meaning_preserving}/{num_correct_and_mp})"
        )
    else:
        print(
            "Robust Flip Rate metrics: Not calculated ('correctness' column missing)."
        )

    # --- Output Summary ---
    print(f"\nOverall Metrics (based on {valid_samples} valid samples):")
    print(f"  Total Flipped Samples: {flipped_samples_count}")
    print(f"  Overall Flip Rate: {overall_flip_rate:.2%}")

    print(f"\nMeaning-Preserving Metrics (Total: {num_meaning_preserving} samples):")
    print(f"  Flipped Samples within Meaning-Preserving: {flips_in_meaning_preserving}")
    print(
        f"  Flip Rate (Meaning-Preserving Flips / All Valid Samples): {flip_rate_mp_vs_all:.2%}"
    )
    # Optional: Flip rate *within* meaning-preserving samples
    # flip_rate_within_mp = flips_in_meaning_preserving / num_meaning_preserving if num_meaning_preserving > 0 else 0
    # print(f"  Flip Rate (within Meaning-Preserving subset): {flip_rate_within_mp:.2%}")

    # Ensure 'Flipped sample examples' section is removed as requested
    # The original code for printing examples is now omitted.

    if export_flipped_csv:
        if (
            "correctness" in df.columns
            and "original_is_correct" in meaning_preserving_df.columns
        ):  # Check prerequisites
            # This is the base for the numerator: meaning-preserving and originally correct
            base_df_for_numerator = meaning_preserving_df[
                meaning_preserving_df["original_is_correct"]
            ]

            # These are the actual samples that make up the numerator
            flipped_samples_to_export_df = base_df_for_numerator[
                base_df_for_numerator["prediction_flipped"] == True
            ].copy()

            if not flipped_samples_to_export_df.empty:
                # Define columns to export - ensure these exist in the source df or are created
                columns_to_select = [
                    "original_samples",
                    "adversarial_samples",
                    "correctness",  # Original ground truth
                    "agreed_labels",  # Should be 0 for these meaning-preserving samples
                    "original_prediction",
                    "adversarial_prediction",
                    "prediction_flipped",  # Should be True for these samples
                ]

                # Ensure all selected columns are present in the dataframe to avoid KeyErrors
                # flipped_samples_to_export_df may not have all columns from the original `df` if it's a slice of a slice.
                # It's safer to re-select from the original `df` using the indices of the flipped samples.
                # The indices of flipped_samples_to_export_df are from the original df.

                export_df = df.loc[
                    flipped_samples_to_export_df.index, columns_to_select
                ].copy()

                export_df.to_csv(export_flipped_csv, index=False)
                print(
                    f"\nExported {len(export_df)} samples (numerator of 'Flip Rate (MP & Correct Originals)') to: {export_flipped_csv}"
                )
            else:
                print(
                    f"\nNo samples to export for 'Flip Rate (MP & Correct Originals)' numerator (path: {export_flipped_csv})."
                )
        else:
            print(
                f"\nCould not export flipped samples to {export_flipped_csv} because 'correctness' or 'original_is_correct' column was missing from relevant dataframe."
            )

    return df


def main():
    # 命令行参数
    parser = argparse.ArgumentParser(
        description="Classify original and adversarial samples using a Llama model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",  # Default to Llama 3
        help="Name of the model to use from Hugging Face Hub (e.g., 'meta-llama/Llama-3.2-1B-Instruct').",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to the LoRA weights directory. If None, uses the base model.",
    )
    parser.add_argument(
        "--data_file",
        default="./data/adversarial_dataset_corrected.csv",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="classification_results.csv",
        help="Path to save the CSV file with classification results.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for model inference."
    )
    parser.add_argument(
        "--token", type=str, default=None, help="Hugging Face API token (if needed)."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process from the data file.",
    )
    parser.add_argument(
        "--export_flipped_csv",
        type=str,
        default=None,
        help="Optional path to save a CSV of (Meaning-Preserving & Correctly Classified Originals) that were flipped.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=CACHE_DIR_TARGET,
        help="Directory for caching Hugging Face models and tokenizers.",
    )

    args = parser.parse_args()

    print(f"Starting classification with model: {args.model_name}")
    if args.lora_path:
        print(f"Applying LoRA weights from: {args.lora_path}")
    print(f"Using cache directory: {args.cache_dir}")

    # 加载数据
    df = load_data(args.data_file)
    if args.max_samples is not None:
        print(f"Processing a maximum of {args.max_samples} samples.")
        df = df.head(args.max_samples)

    # 加载模型和分词器
    tokenizer, model = load_model(
        args.model_name,
        token=args.token,
        lora_path=args.lora_path,
        cache_dir=args.cache_dir,
    )

    # 构建prompts
    print("Constructing prompts for Llama 3 format...")
    (
        original_prompt_messages,
        adversarial_prompt_messages,
        skipped_samples_indices,  # Store the original indices of skipped samples
        valid_samples_count,
    ) = construct_prompts_for_llama3(df, tokenizer)

    # Store skipped indices in df attributes for later use in compare_results
    df.attrs["skipped_samples_indices"] = skipped_samples_indices

    # 分类原始样本
    print("\nClassifying original samples...")
    original_predictions_text = classify_samples(
        tokenizer, model, original_prompt_messages, batch_size=args.batch_size
    )
    original_predictions = [parse_answer(pred) for pred in original_predictions_text]

    # 分类对抗样本
    print("\nClassifying adversarial samples...")
    adversarial_predictions_text = classify_samples(
        tokenizer, model, adversarial_prompt_messages, batch_size=args.batch_size
    )
    adversarial_predictions = [
        parse_answer(pred) for pred in adversarial_predictions_text
    ]

    # 比较结果并保存
    # The df passed here is the original one. compare_results needs to handle skipped samples.
    df_results = compare_results(
        df.copy(),  # Pass a copy to avoid modifying the original df inadvertently before this
        original_predictions,
        adversarial_predictions,
        valid_samples_count,  # This is the count of valid samples
        args.export_flipped_csv,  # Pass the new argument
    )
    # df_results.to_csv(args.output_file, index=False)
    # print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
