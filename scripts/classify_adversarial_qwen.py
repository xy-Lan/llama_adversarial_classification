# scripts/classify_adversarial_qwen.py

import pandas as pd
import torch
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time
from tqdm import tqdm
import numpy as np # Added for np.nan

# --- Start: Set Cache Directories ---
# Ensure all Hugging Face operations use the target cache directory
CACHE_DIR_TARGET = "/mnt/parscratch/users/acc22xl/huggingface"
os.environ["HF_HOME"] = CACHE_DIR_TARGET
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR_TARGET
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR_TARGET
os.environ["HF_METRICS_CACHE"] = CACHE_DIR_TARGET
# --- End: Set Cache Directories ---

def load_data(file_path, nrows=None):
    print(f"Loading data from: {file_path}")
    if nrows:
        print(f"Loading only the first {nrows} rows.")
        df = pd.read_csv(file_path, nrows=nrows)
    else:
        df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows.")
    return df

def construct_chat_prompts(df, tokenizer):
    original_prompts_messages = []
    adversarial_prompts_messages = []
    skipped_indices = [] 
    valid_indices = [] 

    system_message_content = (
        "You are a fact-checking assistant. "
        "Given EVIDENCE and a CLAIM, reply with exactly one token: SUPPORTED or REFUTED. "
        "Do not output anything else."
    )

    for index, row in df.iterrows():
        original_sample = row["original_samples"]
        adversarial_sample = row["adversarial_samples"]

        valid_sample_flag = True
        if not isinstance(original_sample, str):
            # print(f"Skipping sample at original index {index}: Original sample not a string.")
            skipped_indices.append(index)
            valid_sample_flag = False
        else:
            original_sample = original_sample.strip()

        if not isinstance(adversarial_sample, str):
            # print(f"Skipping sample at original index {index}: Adversarial sample not a string.")
            if valid_sample_flag: 
                skipped_indices.append(index)
            valid_sample_flag = False
        else:
            adversarial_sample = adversarial_sample.strip()
        
        if not valid_sample_flag:
            continue

        if "~" not in original_sample or "~" not in adversarial_sample:
            # print(f"Skipping sample at original index {index}: Missing '~' separator.")
            skipped_indices.append(index)
            continue

        original_parts = original_sample.split("~", 1)
        if len(original_parts) != 2 or not original_parts[0].strip() or not original_parts[1].strip():
            # print(f"Skipping sample at original index {index}: Improperly formatted original.")
            skipped_indices.append(index)
            continue

        adversarial_parts = adversarial_sample.split("~", 1)
        if len(adversarial_parts) != 2 or not adversarial_parts[0].strip() or not adversarial_parts[1].strip():
            # print(f"Skipping sample at original index {index}: Improperly formatted adversarial.")
            skipped_indices.append(index)
            continue

        evidence_original, claim_original = original_parts[0].strip(), original_parts[1].strip()
        evidence_adversarial, claim_adversarial = adversarial_parts[0].strip(), adversarial_parts[1].strip()

        original_messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": f"Evidence: {evidence_original}\\nClaim: {claim_original}\\nQuestion: Is the claim supported or refuted by the evidence?\\nAnswer:"}
        ]
        
        adversarial_messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": f"Evidence: {evidence_adversarial}\\nClaim: {claim_adversarial}\\nQuestion: Is the claim supported or refuted by the evidence?\\nAnswer:"}
        ]

        try:
            original_prompt_str = tokenizer.apply_chat_template(
                original_messages, tokenize=False, add_generation_prompt=True
            )
            adversarial_prompt_str = tokenizer.apply_chat_template(
                adversarial_messages, tokenize=False, add_generation_prompt=True
            )
            original_prompts_messages.append(original_prompt_str)
            adversarial_prompts_messages.append(adversarial_prompt_str)
            valid_indices.append(index) # Save original index of valid sample
        except Exception as e:
            # print(f"Error applying chat template for sample at original index {index}: {e}")
            skipped_indices.append(index)

    unique_skipped_count = len(set(skipped_indices))
    valid_samples_count = len(original_prompts_messages)

    print(f"Total samples initially: {len(df)}")
    print(f"Skipped samples (due to formatting or template error): {unique_skipped_count}")
    print(f"Valid samples for processing: {valid_samples_count}")
    
    # Create df_valid containing only the rows for which prompts were successfully generated,
    # preserving original columns needed for compare_results.
    df_valid = df.loc[valid_indices].copy()

    return original_prompts_messages, adversarial_prompts_messages, df_valid, valid_samples_count


def load_qwen_model(model_name="Qwen/Qwen2.5-7B-Instruct", token=None, cache_dir=None, lora_path=None):
    """Loads Qwen model and tokenizer."""
    print(f"Loading Qwen model: {model_name} using cache_dir: {cache_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=token,
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer.pad_token was None, set to tokenizer.eos_token: {tokenizer.eos_token}")

    tokenizer.padding_side = "left"
    print(f"Tokenizer padding_side set to '{tokenizer.padding_side}'")

    model_kwargs = {
        "use_auth_token": token,
        "device_map": "auto",
        "torch_dtype": "auto",
        "trust_remote_code": True,
        "cache_dir": cache_dir
    }
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {gpu_name}")
        if "H100" in gpu_name or "A100" in gpu_name:
             print("Enabling Flash Attention 2 for H100/A100 GPUs.")
             model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        print("No GPU detected, model will run on CPU.")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as e:
        print(f"Error loading model with kwargs {model_kwargs}: {e}")
        print("Falling back to basic loading (might be slower or use more memory)...")
        model_kwargs.pop("attn_implementation", None) 
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=token,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

    if lora_path and os.path.exists(lora_path):
        print(f"Loading LoRA adapter from: {lora_path}")
        try:
            model = PeftModel.from_pretrained(model, lora_path)
            print(f"Successfully loaded LoRA adapter.")
            # Optional: Merge LoRA for inference if desired, e.g., model = model.merge_and_unload()
            # For this script, direct use of PeftModel is usually fine.
        except Exception as e:
            print(f"Error loading LoRA adapter from {lora_path}: {e}")
            print("Proceeding with base model only.")
    elif lora_path:
        print(f"Warning: LoRA path {lora_path} provided but not found. Proceeding with base model only.")

    model.eval()
    device = next(model.parameters()).device
    print(f"Qwen Model '{model_name}'{' with LoRA' if lora_path and os.path.exists(lora_path) else ''} loaded successfully on {device}")
    return tokenizer, model

def classify_samples_qwen(tokenizer, model, prompts, batch_size=8):
    predictions = []
    device = next(model.parameters()).device
    
    if not prompts:
        print("No prompts to classify.")
        return []

    pbar = tqdm(total=(len(prompts) + batch_size - 1) // batch_size, desc="Classifying batches")
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 2048
        ).to(device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10, 
                    do_sample=False,    
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, 
                )
            
            decoded_outputs = []
            for j, output_id_tensor in enumerate(outputs):
                input_length = inputs.input_ids[j].shape[0]
                generated_tokens = output_id_tensor[input_length:]
                decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                decoded_outputs.append(decoded_output)
            predictions.extend(decoded_outputs)
        except Exception as e:
            print(f"Error during generation for batch {i//batch_size}: {e}")
            predictions.extend(["ERROR_GENERATION"] * len(batch_prompts))
        pbar.update(1)
    pbar.close()
    return predictions

def parse_answer(output_text):
    output_lower = output_text.lower().strip()
    if "supported" in output_lower:
        return "SUPPORTED"
    elif "refuted" in output_lower:
        return "REFUTED"
    else:
        if output_lower == "supported":
            return "SUPPORTED"
        if output_lower == "refuted":
            return "REFUTED"
        print(f"Unrecognized answer: '{output_text}' (Defaulting to REFUTED for metrics)")
        return "UNKNOWN" 

def compare_results(df_valid_samples, original_predictions_raw, adversarial_predictions_raw):
    """
    Compares original and adversarial predictions, calculates various metrics.
    Args:
        df_valid_samples (pd.DataFrame): DataFrame filtered to only include the valid samples 
                                         that correspond to the predictions. Must contain 
                                         'correctness' and 'agreed_labels' for full metrics.
        original_predictions_raw (list): List of raw string predictions for original samples.
        adversarial_predictions_raw (list): List of raw string predictions for adversarial samples.
    Returns:
        pd.DataFrame: DataFrame with added prediction and metric columns.
    """
    df = df_valid_samples.copy() # Work on a copy
    
    if len(df) == 0:
        print("No valid samples to compare after filtering. Returning empty DataFrame.")
        return df

    # Parse predictions
    original_parsed = [parse_answer(p) for p in original_predictions_raw]
    adversarial_parsed = [parse_answer(p) for p in adversarial_predictions_raw]

    # Add raw and parsed predictions
    if len(df) != len(original_parsed) or len(df) != len(adversarial_parsed):
        print(f"Critical Error: Length mismatch between df_valid_samples ({len(df)}) and predictions. Cannot proceed with compare_results.")
        # Optionally, return df with what's available or raise error
        df["original_prediction_raw"] = pd.Series(original_predictions_raw).reindex(df.index)
        df["adversarial_prediction_raw"] = pd.Series(adversarial_predictions_raw).reindex(df.index)
        df["original_prediction_parsed"] = pd.Series(original_parsed).reindex(df.index)
        df["adversarial_prediction_parsed"] = pd.Series(adversarial_parsed).reindex(df.index)
        return df # Or raise an exception
    
    df["original_prediction_raw"] = original_predictions_raw
    df["adversarial_prediction_raw"] = adversarial_predictions_raw
    df["original_prediction_parsed"] = original_parsed
    df["adversarial_prediction_parsed"] = adversarial_parsed
    
    # --- Clean Accuracy ---
    num_correct_original_samples_valid_gt = 0
    num_samples_with_valid_gt = 0
    clean_accuracy = 0.0

    if "correctness" in df.columns:
        df["mapped_correctness"] = (
            df["correctness"]
            .astype(str) 
            .str.upper()
            .replace({"SUPPORTS": "SUPPORTED", "REFUTES": "REFUTED"})
        )
        valid_ground_truth_labels = ["SUPPORTED", "REFUTED"]
        df["is_valid_ground_truth"] = df["mapped_correctness"].isin(valid_ground_truth_labels)
        
        evaluable_gt_mask = df["original_prediction_parsed"].notna() & df["is_valid_ground_truth"]
        num_samples_with_valid_gt = evaluable_gt_mask.sum()

        df["original_is_correct"] = False 
        if num_samples_with_valid_gt > 0: # Avoid .loc if mask is all False
            df.loc[evaluable_gt_mask, "original_is_correct"] = (
                df.loc[evaluable_gt_mask, "original_prediction_parsed"] == df.loc[evaluable_gt_mask, "mapped_correctness"]
            )
        num_correct_original_samples_valid_gt = df["original_is_correct"].sum()
        
        clean_accuracy = (
            num_correct_original_samples_valid_gt / num_samples_with_valid_gt
            if num_samples_with_valid_gt > 0 else 0.0
        )
        print(
            f"\nClean Accuracy (on original samples with valid ground truth): {clean_accuracy:.2%} ({num_correct_original_samples_valid_gt}/{num_samples_with_valid_gt})"
        )
    else:
        print("\nWarning: 'correctness' column not found. Skipping Clean Accuracy and related metrics.")
        df["original_is_correct"] = False 

    # --- Flip Rate ---
    evaluable_flip_mask = df["original_prediction_parsed"].notna() & df["adversarial_prediction_parsed"].notna()
    df["prediction_flipped"] = False 
    if evaluable_flip_mask.sum() > 0: # Avoid .loc if mask is all False
        df.loc[evaluable_flip_mask, "prediction_flipped"] = (
            df.loc[evaluable_flip_mask, "original_prediction_parsed"] != df.loc[evaluable_flip_mask, "adversarial_prediction_parsed"]
        )
    
    num_evaluable_for_flip = evaluable_flip_mask.sum() 
    flipped_samples_count = df["prediction_flipped"].sum()
    overall_flip_rate = flipped_samples_count / num_evaluable_for_flip if num_evaluable_for_flip > 0 else 0.0

    # --- Meaning-Preserving Metrics ---
    # Default to all preserving if 'agreed_labels' is missing.
    df_temp_agreed_labels = df.get("agreed_labels", pd.Series([0] * len(df), index=df.index)) 
    
    meaning_preserving_mask = (df_temp_agreed_labels == 0) & evaluable_flip_mask
    meaning_preserving_df = df[meaning_preserving_mask].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    num_meaning_preserving_evaluable = len(meaning_preserving_df)
    flips_in_meaning_preserving = meaning_preserving_df["prediction_flipped"].sum()
    
    flip_rate_mp_vs_all_evaluable = flips_in_meaning_preserving / num_evaluable_for_flip if num_evaluable_for_flip > 0 else 0.0

    # --- Robust Flip Rate (Targeted MP Flip Rate) ---
    num_correct_and_mp = 0
    flips_in_correct_and_mp = 0
    robust_flip_rate = 0.0
    mp_flip_rate_targeted = np.nan # Use np.nan

    if "correctness" in df.columns and "original_is_correct" in df.columns : # original_is_correct must exist
        correct_and_meaning_preserving_mask = meaning_preserving_mask & df["original_is_correct"]
        correct_and_meaning_preserving_df = df[correct_and_meaning_preserving_mask].copy()
        
        num_correct_and_mp = len(correct_and_meaning_preserving_df)
        flips_in_correct_and_mp = correct_and_meaning_preserving_df["prediction_flipped"].sum()
        
        robust_flip_rate = (
            flips_in_correct_and_mp / num_correct_and_mp if num_correct_and_mp > 0 else 0.0
        )
        
        mp_flip_rate_targeted = (
            flips_in_meaning_preserving / num_correct_and_mp 
            if num_correct_and_mp > 0 
            else np.nan
        )

        print(f"Meaning-Preserving & Correctly Classified Original Samples (evaluable for flip): {num_correct_and_mp}")
        print(f"Flip Rate (within Meaning-Preserving & Correctly Classified Originals): {robust_flip_rate:.2%} ({flips_in_correct_and_mp}/{num_correct_and_mp})")
        print(f"Targeted MP Flip Rate (All MP Flips / MP & Correctly Classified Originals): {mp_flip_rate_targeted:.2%} ({flips_in_meaning_preserving}/{num_correct_and_mp})")
    else:
        print("Robust Flip Rate metrics: Not calculated ('correctness' column missing or 'original_is_correct' not generated).")

    # --- Output Summary ---
    print(f"\nOverall Metrics (based on {num_evaluable_for_flip} samples with valid original and adversarial predictions):")
    print(f"  Total Flipped Samples: {flipped_samples_count}")
    print(f"  Overall Flip Rate: {overall_flip_rate:.2%}")

    print(f"\nMeaning-Preserving Metrics (based on {num_meaning_preserving_evaluable} meaning-preserving samples with valid predictions):")
    print(f"  Flipped Samples within these Meaning-Preserving ones: {flips_in_meaning_preserving}")
    print(f"  Flip Rate (Meaning-Preserving Flips / All Evaluable Samples): {flip_rate_mp_vs_all_evaluable:.2%}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Classify original and adversarial samples using Qwen model.")
    parser.add_argument("--data_file", type=str, default="./data/adversarial_dataset_corrected.csv", help="Path to the CSV data file.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model ID for Qwen.")
    parser.add_argument("--output_file", type=str, default="qwen_classification_results.csv", help="Path to save the detailed results CSV.")
    parser.add_argument("--summary_file", type=str, default="qwen_classification_summary.txt", help="Path to save the summary of results.")
    parser.add_argument("--auth_token", type=str, default=None, help="Hugging Face authentication token.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for model inference.")
    parser.add_argument("--nrows", type=int, default=None, help="Number of rows to load from the CSV for quick testing.")
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR_TARGET, help="Cache directory for Hugging Face models and tokenizers.")
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to the LoRA weights directory. If None, uses the base model.",
    )
    parser.add_argument(
        "--export_flipped_csv",
        type=str,
        default=None,
        help="Optional path to save a CSV of (Meaning-Preserving & Correctly Classified Originals) that were flipped.",
    )

    args = parser.parse_args()

    print(f"Starting classification with Qwen model: {args.model_name}")
    print(f"Data file: {args.data_file}")
    if args.cache_dir:
        print(f"Using cache directory: {args.cache_dir}")

    start_time_total = time.time()

    df_full = load_data(args.data_file, nrows=args.nrows)
    
    tokenizer, model = load_qwen_model(args.model_name, token=args.auth_token, cache_dir=args.cache_dir, lora_path=args.lora_path)

    original_prompts, adversarial_prompts, df_valid, valid_samples_count = construct_chat_prompts(df_full, tokenizer)

    if not valid_samples_count:
        print("No valid samples to process after prompt construction. Exiting.")
        return

    print("\nClassifying original samples...")
    original_predictions = classify_samples_qwen(tokenizer, model, original_prompts, batch_size=args.batch_size)

    print("\nClassifying adversarial samples...")
    adversarial_predictions = classify_samples_qwen(tokenizer, model, adversarial_prompts, batch_size=args.batch_size)

    # Pass df_valid (which should contain original 'correctness', 'agreed_labels' etc.)
    df_results = compare_results(
        df_valid, # df_valid already contains only valid samples with original columns
        original_predictions, 
        adversarial_predictions
    )

    df_results.to_csv(args.output_file, index=False)
    print(f"\nDetailed results saved to: {args.output_file}")

    # --- Export specific flipped samples if requested ---
    if args.export_flipped_csv:
        # `df_results` is the main DataFrame, equivalent to `df` in the target example.
        # Prerequisites for the logic: 'correctness', 'original_is_correct', 'agreed_labels', 'prediction_flipped'
        # and columns for final selection must be in df_results.
        
        columns_needed_for_export_logic = [
            "correctness", "original_is_correct", "agreed_labels", "prediction_flipped",
            "original_samples", "adversarial_samples", 
            "original_prediction_parsed", "adversarial_prediction_parsed"
        ]

        if all(col in df_results.columns for col in columns_needed_for_export_logic):
            # Step 1: Filter for meaning-preserving samples (equivalent to target's `meaning_preserving_df`)
            # In Qwen script, agreed_labels == 0 means meaning-preserving.
            meaning_preserving_df = df_results[df_results["agreed_labels"] == 0]
            
            # Step 2: From meaning-preserving, filter for originally correct samples
            # (equivalent to target's `base_df_for_numerator`)
            # 'original_is_correct' is a boolean column in df_results.
            base_df_for_numerator = meaning_preserving_df[
                meaning_preserving_df["original_is_correct"] # Filter by boolean column directly
            ]
            
            # Step 3: From these, filter for samples where prediction was flipped
            # (equivalent to target's `flipped_samples_to_export_df`)
            # 'prediction_flipped' is a boolean column in df_results.
            flipped_samples_to_export_df = base_df_for_numerator[
                base_df_for_numerator["prediction_flipped"] == True # Explicitly check True
            ].copy() # .copy() as in target

            if not flipped_samples_to_export_df.empty:
                columns_to_select = [
                    "original_samples", 
                    "adversarial_samples", 
                    "correctness", # Original ground truth
                    "agreed_labels", 
                    "original_prediction_parsed", # Mapped from original_prediction in target
                    "adversarial_prediction_parsed",# Mapped from adversarial_prediction in target
                    "prediction_flipped"
                ]
                
                # Re-select from the original `df_results` using the indices
                # from `flipped_samples_to_export_df`.
                export_df = df_results.loc[flipped_samples_to_export_df.index, columns_to_select].copy()
                
                try:
                    export_df.to_csv(args.export_flipped_csv, index=False)
                    print(f"\nExported {len(export_df)} samples (numerator of 'Flip Rate (MP & Correct Originals)') to: {args.export_flipped_csv}")
                except Exception as e:
                    print(f"Error exporting specific flipped samples: {e}") # General file op error
            else:
                print(f"\nNo samples to export for 'Flip Rate (MP & Correct Originals)' numerator (path: {args.export_flipped_csv}).")
        else:
            # Use the exact error message from the target script for the prerequisite failure.
            print(f"\nCould not export flipped samples to {args.export_flipped_csv} because 'correctness' or 'original_is_correct' column was missing from relevant dataframe.")

    # --- Extract metrics from df_results for summary ---
    summary_valid_samples = len(df_results)
    summary_original_supported = (df_results["original_prediction_parsed"] == "SUPPORTED").sum() if "original_prediction_parsed" in df_results else 0
    summary_original_refuted = (df_results["original_prediction_parsed"] == "REFUTED").sum() if "original_prediction_parsed" in df_results else 0
    
    summary_successful_attacks = 0
    if "original_prediction_parsed" in df_results and "adversarial_prediction_parsed" in df_results:
        summary_successful_attacks = (
            (df_results["original_prediction_parsed"] == "SUPPORTED") & 
            (df_results["adversarial_prediction_parsed"] == "REFUTED")
        ).sum()

    summary_asr = (summary_successful_attacks / summary_original_supported) * 100 if summary_original_supported > 0 else 0

    with open(args.summary_file, "w") as f:
        f.write("--- Qwen Classification Summary ---\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Data File: {args.data_file}\n")
        f.write(f"Processed {args.nrows if args.nrows else 'all'} initial rows, resulting in {summary_valid_samples} valid samples for classification.\n")
        f.write(f"Total valid samples for comparison: {summary_valid_samples}\n") # From new compare_results df
        
        # Metrics from df_results (check if columns exist before accessing)
        if "original_prediction_parsed" in df_results:
            f.write(f"Number of original samples predicted as SUPPORTED: {summary_original_supported}\n")
            f.write(f"Number of original samples predicted as REFUTED: {summary_original_refuted}\n")

        if "correctness" in df_results and "original_is_correct" in df_results:
             num_correct_orig = df_results["original_is_correct"].sum()
             num_valid_gt = df_results["is_valid_ground_truth"].sum() if "is_valid_ground_truth" in df_results else 0
             clean_acc = num_correct_orig / num_valid_gt if num_valid_gt > 0 else 0
             f.write(f"Clean Accuracy: {clean_acc:.2%} ({num_correct_orig}/{num_valid_gt})\n")
        
        if "prediction_flipped" in df_results:
            total_flips = df_results["prediction_flipped"].sum()
            num_eval_flips = (df_results["original_prediction_parsed"].notna() & df_results["adversarial_prediction_parsed"].notna()).sum()
            overall_flip_r = total_flips / num_eval_flips if num_eval_flips > 0 else 0
            f.write(f"Overall Flip Rate: {overall_flip_r:.2%}\n")
        
        f.write(f"Successful attacks (Original SUPPORTED -> Adversarial REFUTED): {summary_successful_attacks}\n")
        f.write(f"Attack Success Rate (ASR) among 'originally SUPPORTED' samples: {summary_asr:.2f}%\n")
    print(f"Summary saved to: {args.summary_file}")

    end_time_total = time.time()
    print(f"Total execution time: {end_time_total - start_time_total:.2f} seconds")

if __name__ == "__main__":
    main()