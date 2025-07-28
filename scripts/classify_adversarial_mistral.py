#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/classify_adversarial_mistral.py

import pandas as pd
import torch
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from tqdm import tqdm
from peft import PeftModel
import sys

# Correct the target cache directory path
CACHE_DIR_TARGET = "/mnt/parscratch/users/acc22xl/huggingface_cache/"
os.environ["HF_HOME"] = CACHE_DIR_TARGET
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR_TARGET
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR_TARGET
os.environ["HF_METRICS_CACHE"] = CACHE_DIR_TARGET # Though less common, update for consistency


def load_data(file_path):
    # 与原CPU版本保持一致，只加载前50行数据
    df = pd.read_csv(file_path)
    # df = pd.read_csv(file_path, nrows=50)
    return df


def construct_prompts_for_mistral(df, tokenizer):
    original_prompt_messages = []
    adversarial_prompt_messages = []
    skipped_samples = []  # 记录被剔除的样本索引

    # Mistral system message (can be adjusted if needed)
    # The tokenizer.apply_chat_template will incorporate this correctly for Mistral.
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
            f"Evidence: {evidence_original.strip()}\n"
            f"Claim: {claim_original.strip()}\n"
            "Question: Is the claim supported or refuted by the evidence?\n"
            "Answer:"
        )

        user_message_adversarial_content = (
            f"Evidence: {evidence_adversarial.strip()}\n"
            f"Claim: {claim_adversarial.strip()}\n"
            "Question: Is the claim supported or refuted by the evidence?\n"
            "Answer:"
        )

        # Mistral chat format (using messages list, apply_chat_template handles the specific [INST] tags)
        original_messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_original_content},
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

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name} - {gpu.total_memory/1e9:.2f} GB")
        if any(
            gpu_type in gpu.name for gpu_type in ["A100", "H100", "A6000", "RTX 4090"]
        ):
            print("High-end GPU detected, enabling TF32 precision")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        mem_gb = gpu.total_memory / 1e9
        use_4bit = mem_gb < 24 
        use_8bit = 24 <= mem_gb < 40 and not use_4bit
        print(f"Memory optimization: 4bit={use_4bit}, 8bit={use_8bit}") # Note: Quantization args not used in this script version
    else:
        print("GPU not available, using CPU")
        use_4bit = False
        use_8bit = False

    model_kwargs = {
        "use_auth_token": token, # "token" is the more current HF arg name
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
        "cache_dir": cache_dir,
        "trust_remote_code": True # Often needed for instruct/chat models
    }

    print(f"Loading tokenizer for base model: {model_name} using cache: {cache_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=token, # use "token"
        cache_dir=cache_dir, 
        trust_remote_code=True,
        model_max_length=32768  # <--- Explicitly set model_max_length
    )

    # Mistral models typically expect left padding for batched generation.
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        print("pad_token is None. Setting pad_token = eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure model config reflects pad_token_id if it's set on model init
    # model.config.pad_token_id = tokenizer.pad_token_id (can be set after model load)


    try:
        print(f"Loading base model: {model_name} with config: {model_kwargs}")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as e:
        print(f"Error loading base model with specified dtype/config: {e}")
        print("Falling back to basic loading for base model (FP32 default if GPU available and not BF16)...")
        # Fallback: remove torch_dtype or set to float32 if bfloat16 failed
        model_kwargs.pop("torch_dtype", None)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            token=token,
            device_map="auto", # keep auto device_map
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        # Manually move to cuda if not done by device_map and GPU is available
        if torch.cuda.is_available() and next(model.parameters()).device.type == 'cpu':
            model = model.to("cuda")


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

    model.eval()
    device = next(model.parameters()).device
    print(f"Model (final) loaded successfully on {device}")

    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        print(
            f"Setting model.config.pad_token_id to tokenizer.pad_token_id ({tokenizer.pad_token_id})"
        )
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


def classify_samples(tokenizer, model, prompt_messages_list, batch_size=8):
    predictions = []
    device = next(model.parameters()).device
    global_debug_printed = {'printed': False}  # 用字典保证引用一致

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        mem_gb = gpu.total_memory / 1e9
        suggested_batch = max(
            1, min(int(mem_gb // 5), 32)
        )
        if batch_size == 1 and len(prompt_messages_list) > 1 :
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

    total_batches = (len(prompt_messages_list) + batch_size - 1) // batch_size
    pbar = tqdm(total=total_batches, desc="Processing batches")

    start_time = time.time()
    for i in range(0, len(prompt_messages_list), batch_size):
        batch_messages = prompt_messages_list[i : i + batch_size]
        batch_start = time.time()

        batch_prompts_formatted = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in batch_messages
        ]

        try:
            inputs = tokenizer(
                batch_prompts_formatted, padding=True, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            input_lengths = inputs.input_ids.shape[1]
            for j, output_sequence in enumerate(outputs):
                generated_tokens = output_sequence[input_lengths:]
                generated_text = tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                ).strip()
                prediction = parse_answer(generated_text)
                predictions.append(prediction)

                if i == 0 and j < 3:
                    print(f"  Sample {j+1}:")
                    printable_prompt = batch_prompts_formatted[j]
                    if len(printable_prompt) > 300:
                        printable_prompt = printable_prompt[:150] + "..." + printable_prompt[-150:]
                    print(f"    Prompt (Mistral formatted): {printable_prompt}")
                    print(f"    Generated: {generated_text}")
                    print(f"    Prediction: {prediction}")

            batch_time = time.time() - batch_start
            samples_per_sec = len(batch_prompts_formatted) / batch_time if batch_time > 0 else float('inf')
            pbar.set_postfix(
                samples_per_sec=f"{samples_per_sec:.1f}",
                batch_time=f"{batch_time:.2f}s",
            )

        except Exception as e:
            print(f"Error in batch processing (batch starting at index {i}): {e}")
            print("Attempting to process samples in this batch individually...")
            for k, individual_messages in enumerate(batch_messages):
                try: # Outer try-except for apply_chat_template and subsequent steps
                    prompt_formatted = tokenizer.apply_chat_template(
                        individual_messages, tokenize=False, add_generation_prompt=True
                    )
                    print(f"  Processing individual prompt {i+k+1} (fallback)")

                    try: # Inner try-except for tokenizer() and model.generate()
                        if not global_debug_printed['printed']: # Only print for the first attempt that might fail
                            print(f"DEBUG: About to call tokenizer for sample {i+k+1}")
                            print(f"DEBUG: tokenizer.model_max_length = {tokenizer.model_max_length}")
                            print(f"DEBUG: type(tokenizer.model_max_length) = {type(tokenizer.model_max_length)}")
                            print(f"DEBUG: tokenizer.vocab_size = {tokenizer.vocab_size}")
                            print(f"DEBUG: len(prompt_formatted) = {len(prompt_formatted)}")
                            print(f"DEBUG: prompt_formatted[:500] = {prompt_formatted[:500]}") # Print start of prompt

                        inputs_single = tokenizer(
                            prompt_formatted,
                            return_tensors="pt",
                            truncation=True,
                            max_length=tokenizer.model_max_length
                        ).to(device)

                        with torch.no_grad():
                            outputs_single = model.generate(
                                **inputs_single,
                                max_new_tokens=10,
                                do_sample=False,
                                temperature=None,
                                top_p=None,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            )

                        generated_tokens_single = outputs_single[0, inputs_single.input_ids.shape[1]:]
                        generated_text_single = tokenizer.decode(
                            generated_tokens_single, skip_special_tokens=True
                        ).strip()
                        prediction_single = parse_answer(generated_text_single)
                        predictions.append(prediction_single)
                        print(f"    Fallback Sample {i+k+1} - Generated: {generated_text_single}, Prediction: {prediction_single}")

                    except Exception as inner_e: # Catches errors from tokenizer() or model.generate()
                        print(f"  Error in single sample processing (fallback for sample {i+k+1}): {inner_e}")
                        if not global_debug_printed['printed']:
                            print("==== FIRST ERROR DETAILS (tokenizer or generation failed) ====")
                            print(f"Sample index: {i+k+1}")
                            print(f"Prompt length (characters): {len(prompt_formatted)}")
                            print("Prompt content that was being processed:")
                            print(prompt_formatted)
                            print("---")
                            if 'inputs_single' in locals() and inputs_single is not None:
                                print("inputs_single was created (tokenizer likely succeeded, error might be in generate?):")
                                try:
                                    print("  input_ids:", inputs_single["input_ids"])
                                    print("  max token id:", inputs_single["input_ids"].max().item())
                                    print("  min token id:", inputs_single["input_ids"].min().item())
                                except Exception as e_print_ids:
                                    print(f"  (Error printing input_ids details: {e_print_ids})")
                            else:
                                print("inputs_single was NOT created (tokenizer() call itself likely failed with the error above).")
                            print("---")
                            # Ensure model_max_length is printed again here if it wasn't due to early exit logic
                            if tokenizer is not None:
                                print(f"VERIFY: tokenizer.model_max_length at time of error log = {tokenizer.model_max_length}")
                            global_debug_printed['printed'] = True
                            print("Exiting after first error to capture debug info.")
                            sys.exit(1)
                        predictions.append("UNKNOWN")

                except Exception as outer_e: # Catches errors from apply_chat_template
                    print(f"  Error in prompt construction (fallback for sample {i+k+1}): {outer_e}")
                    if not global_debug_printed['printed']:
                        print("==== FIRST ERROR DETAILS (prompt construction failed) ====")
                        print(f"Sample index: {i+k+1}")
                        print("Individual messages that caused the error:")
                        print(f"  individual_messages: {individual_messages}")
                        print("---")
                        global_debug_printed['printed'] = True
                        print("Exiting after first error to capture debug info.")
                        sys.exit(1)
                    predictions.append("UNKNOWN")
        pbar.update(1)
    pbar.close()

    total_time = time.time() - start_time
    avg_samples_per_sec = len(prompt_messages_list) / total_time if total_time > 0 else float('inf')
    print(
        f"Total processing time: {total_time:.2f}s, Average speed: {avg_samples_per_sec:.1f} samples/sec"
    )

    if predictions: 
        supported_count = predictions.count("SUPPORTED")
        refuted_count = predictions.count("REFUTED")
        unknown_count = predictions.count("UNKNOWN")
        total_preds = len(predictions)

        print(f"\nPrediction distribution ({total_preds} predictions):")
        print(
            f"  SUPPORTED: {supported_count} ({supported_count/total_preds*100:.1f}%)"
        )
        print(f"  REFUTED: {refuted_count} ({refuted_count/total_preds*100:.1f}%)")
        if unknown_count > 0 or total_preds == 0: 
            print(f"  UNKNOWN: {unknown_count} ({unknown_count/total_preds*100 if total_preds > 0 else 0:.1f}%)")
    else:
        print("No predictions were made.")

    return predictions


def parse_answer(output_text):
    # This function is designed to be robust for "SUPPORTED" or "REFUTED"
    # Mistral instruct models are generally good at following instructions for one-word answers.
    
    cleaned_output_upper = output_text.strip().upper()

    # Check for direct keywords first
    if "SUPPORTED" in cleaned_output_upper:
        # Prioritize if it's the first word or the only word, to avoid "NOT SUPPORTED"
        words = cleaned_output_upper.split()
        if words and words[0] == "SUPPORTED":
            return "SUPPORTED"
    if "REFUTED" in cleaned_output_upper:
        words = cleaned_output_upper.split()
        if words and words[0] == "REFUTED":
            return "REFUTED"

    # If not clearly "SUPPORTED" or "REFUTED" as the primary part of the answer.
    # Defaulting to REFUTED might be safer in some contexts, but UNKNOWN is more accurate for failed parsing
    # print(f"Unrecognized answer: '{output_text}' (defaulting to UNKNOWN)")
    # Check if the exact word is present, even if not first
    if "SUPPORTED" in cleaned_output_upper: return "SUPPORTED" # Re-check without word order constraint
    if "REFUTED" in cleaned_output_upper: return "REFUTED"
    
    print(f"Unrecognized answer: '{output_text}' (Defaulting to REFUTED for metrics)")
    return "UNKNOWN"


def compare_results(df, original_predictions, adversarial_predictions, valid_samples_count, export_flipped_csv=None, skipped_samples_indices=None):
    if skipped_samples_indices is None:
        skipped_samples_indices = []

    # Create columns for predictions, initially empty or with a placeholder
    df["original_prediction"] = pd.NA
    df["adversarial_prediction"] = pd.NA

    processed_indices = [idx for idx in df.index if idx not in skipped_samples_indices]

    # Ensure lengths match before assigning
    if len(processed_indices) == len(original_predictions):
        df.loc[processed_indices, "original_prediction"] = original_predictions
    else:
        print(f"Warning: Mismatch in length between processed indices ({len(processed_indices)}) and original_predictions ({len(original_predictions)}). Alignment may be incorrect.")
        # Fallback or error handling if lengths don't match processed_indices
        if len(df) == len(original_predictions): # If df was pre-filtered
             df["original_prediction"] = original_predictions


    if len(processed_indices) == len(adversarial_predictions):
        df.loc[processed_indices, "adversarial_prediction"] = adversarial_predictions
    else:
        print(f"Warning: Mismatch in length between processed indices ({len(processed_indices)}) and adversarial_predictions ({len(adversarial_predictions)}). Alignment may be incorrect.")
        if len(df) == len(adversarial_predictions): # If df was pre-filtered
            df["adversarial_prediction"] = adversarial_predictions


    df["comparison_result"] = "Not Run" 

    if "correctness" not in df.columns:
        print(
            "Warning: 'correctness' column not found. Skipping Clean Accuracy and related metrics."
        )
        clean_accuracy = 0
        df["original_is_correct"] = False 
        num_correct_original_samples_valid_gt = 0
    else:
        df["mapped_correctness"] = (
            df["correctness"]
            .astype(str).str.upper() # Ensure string type before .str accessor
            .replace(
                {
                    "SUPPORTS": "SUPPORTED",
                    "REFUTES": "REFUTED",
                }
            )
        )
        valid_ground_truth_labels = ["SUPPORTED", "REFUTED"]
        df["is_valid_ground_truth"] = df["mapped_correctness"].isin(
            valid_ground_truth_labels
        )

        num_samples_with_valid_gt = df["is_valid_ground_truth"].sum()

        # Ensure predictions are strings for comparison
        df["original_prediction"] = df["original_prediction"].astype(str)

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

    # Filter out rows where predictions might be NA (e.g. due to earlier errors or skips not caught by processed_indices)
    # before calculating prediction_flipped
    valid_pred_mask = df["original_prediction"].notna() & df["adversarial_prediction"].notna()
    
    df["prediction_flipped"] = False # Initialize column
    df.loc[valid_pred_mask, "prediction_flipped"] = df.loc[valid_pred_mask, "original_prediction"] != df.loc[valid_pred_mask, "adversarial_prediction"]


    # valid_samples_count is the number of samples for which prompts were built
    # Use this for flip rate denominator as it represents successfully processed pairs
    flipped_samples_count = df.loc[valid_pred_mask, "prediction_flipped"].sum()
    
    # Ensure valid_samples_count is used as the denominator for overall flip rate
    # This count comes from construct_prompts function and represents pairs that went into classification
    if valid_samples_count > 0 :
        overall_flip_rate = flipped_samples_count / valid_samples_count
    else:
        overall_flip_rate = 0


    meaning_preserving_mask = df["agreed_labels"] == 0
    meaning_preserving_df = df[meaning_preserving_mask & valid_pred_mask].copy() # Also ensure predictions exist
    num_meaning_preserving_evaluable = len(meaning_preserving_df) # MP samples with valid predictions
    
    flips_in_meaning_preserving = meaning_preserving_df["prediction_flipped"].sum()

    # This rate is "Flips in Meaning-Preserving / All Valid Samples (from prompt construction)"
    flip_rate_mp_vs_all = (
        flips_in_meaning_preserving / valid_samples_count if valid_samples_count > 0 else 0
    )

    if "correctness" in df.columns:
        correct_and_meaning_preserving_df = meaning_preserving_df[
            meaning_preserving_df["original_is_correct"] # original_is_correct already implies valid_pred_mask due to its calculation
        ]
        num_correct_and_mp_evaluable = len(correct_and_meaning_preserving_df)

        flips_in_correct_and_mp = correct_and_meaning_preserving_df[
            "prediction_flipped"
        ].sum()

        robust_flip_rate = ( # Flips in (Correct & MP) / Count of (Correct & MP)
            flips_in_correct_and_mp / num_correct_and_mp_evaluable
            if num_correct_and_mp_evaluable > 0
            else 0
        )

        print(
            f"Meaning-Preserving & Correctly Classified Original Samples (evaluable): {num_correct_and_mp_evaluable}"
        )
        print(
            f"Flip Rate (for Meaning-Preserving & Correctly Classified Originals): {robust_flip_rate:.2%} ({flips_in_correct_and_mp}/{num_correct_and_mp_evaluable})"
        )
        
        if num_correct_and_mp_evaluable > 0: # Denominator for targeted MP flip rate
            mp_flip_rate_targeted = flips_in_meaning_preserving / num_correct_and_mp_evaluable
        else:
            mp_flip_rate_targeted = float("nan")
        print(
            f"Targeted MP Flip Rate (All Meaning-Preserving Flips / MP & Correctly Classified Originals): {mp_flip_rate_targeted:.2%} ({flips_in_meaning_preserving}/{num_correct_and_mp_evaluable})"
        )

    else:
        print(
            "Robust Flip Rate metrics: Not calculated ('correctness' column missing)."
        )
        robust_flip_rate = float('nan') # Define for consistency if correctness is missing


    print(f"\nOverall Metrics (based on {valid_samples_count} valid sample pairs from prompt construction):")
    print(f"  Total Flipped Pairs (where both original/adversary were predicted): {flipped_samples_count}")
    print(f"  Overall Flip Rate: {overall_flip_rate:.2%}")

    print(f"\nMeaning-Preserving Metrics (Total evaluable: {num_meaning_preserving_evaluable} pairs with agreed_labels=0 and valid predictions):")
    print(f"  Flipped Pairs within Meaning-Preserving (evaluable): {flips_in_meaning_preserving}")
    print(
        f"  Flip Rate (Meaning-Preserving Flips / All Valid Sample Pairs): {flip_rate_mp_vs_all:.2%}"
    )
    
    # Flip rate *within* the meaning-preserving subset that was evaluable
    flip_rate_within_mp_evaluable = flips_in_meaning_preserving / num_meaning_preserving_evaluable if num_meaning_preserving_evaluable > 0 else 0
    print(f"  Flip Rate (within Meaning-Preserving evaluable subset): {flip_rate_within_mp_evaluable:.2%}")


    if export_flipped_csv:
        # We want to export samples that are:
        # 1. Meaning-preserving (agreed_labels == 0)
        # 2. Originally classified correctly (original_is_correct == True)
        # 3. Prediction actually flipped (prediction_flipped == True)
        if "correctness" in df.columns and "original_is_correct" in df.columns:
            # Start with meaning-preserving that were originally correct
            base_df_for_export = df[
                (df["agreed_labels"] == 0) & 
                (df["original_is_correct"] == True) &
                (valid_pred_mask) # Ensure they had valid predictions
            ].copy()
            
            # From this subset, select those that actually flipped
            flipped_samples_to_export_df = base_df_for_export[
                base_df_for_export["prediction_flipped"] == True
            ].copy()

            if not flipped_samples_to_export_df.empty:
                columns_to_select = [
                    "original_samples", 
                    "adversarial_samples", 
                    "correctness", 
                    "agreed_labels", 
                    "original_prediction", 
                    "adversarial_prediction",
                    "prediction_flipped",
                    "mapped_correctness", # Helpful for debugging
                    "original_is_correct" # Helpful for debugging
                ]
                # Ensure all selected columns exist; add if missing for safety
                for col in columns_to_select:
                    if col not in flipped_samples_to_export_df.columns:
                        flipped_samples_to_export_df[col] = pd.NA # Add as NA if missing

                export_df = flipped_samples_to_export_df[columns_to_select]
                
                export_df.to_csv(export_flipped_csv, index=False)
                print(f"\nExported {len(export_df)} samples (numerator of 'Robust Flip Rate') to: {export_flipped_csv}")
            else:
                print(f"\nNo samples to export for Robust Flip Rate numerator (path: {export_flipped_csv}). (Meaning-preserving, originally correct, and flipped).")
        else:
            print(f"\nCould not export flipped samples to {export_flipped_csv}: 'correctness' or 'original_is_correct' column processing issue.")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Classify original and adversarial samples using a Mistral Instruct model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3", 
        help="Name of the Mistral Instruct model to use from Hugging Face Hub.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to the LoRA weights directory. If None, uses the base model.",
    )
    parser.add_argument(
        "--data_file",
        default="./data/adversarial_dataset_corrected.csv", # Ensure this path is correct
        help="Path to dataset CSV file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="classification_results_mistral.csv",
        help="Path to save the CSV file with classification results.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for model inference."
    )
    parser.add_argument(
        "--token", type=str, default="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", help="Hugging Face API token (if needed for gated models, though Mistral-7B-Instruct v0.3 is open)."
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
        default=None, # e.g., "flipped_robust_mistral.csv"
        help="Optional path to save a CSV of (Meaning-Preserving & Correctly Classified Originals) that were flipped.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=CACHE_DIR_TARGET,
        help="Directory for caching Hugging Face models and tokenizers.",
    )

    args = parser.parse_args()

    print(f"Starting classification with Mistral model: {args.model_name}")
    if args.lora_path:
        print(f"Applying LoRA weights from: {args.lora_path}")
    print(f"Using data file: {args.data_file}")
    print(f"Using cache directory: {args.cache_dir}")
    if args.max_samples is not None:
        print(f"Processing a maximum of {args.max_samples} samples.")


    df = load_data(args.data_file)
    if args.max_samples is not None:
        df = df.head(args.max_samples)

    tokenizer, model = load_model(
        args.model_name, token=args.token, lora_path=args.lora_path, cache_dir=args.cache_dir
    )


    print("Constructing prompts for Mistral format...")
    (
        original_prompt_messages,
        adversarial_prompt_messages,
        skipped_samples_indices, 
        valid_samples_count, # This is the count of pairs for which prompts were successfully constructed
    ) = construct_prompts_for_mistral(df, tokenizer)

    # Filter df to only include rows for which prompts were constructed successfully BEFORE classification
    # This ensures that classify_samples and compare_results operate on the same set of samples.
    # However, original_predictions and adversarial_predictions will have length = valid_samples_count.
    # compare_results needs the original df to fill in results correctly.
    # So, we pass skipped_samples_indices to compare_results.

    # Store skipped indices in df attributes for potential use, though direct passing is better
    # df.attrs["skipped_samples_indices"] = skipped_samples_indices


    if not original_prompt_messages:
        print("No valid original samples to classify after prompt construction. Exiting.")
        return
    if not adversarial_prompt_messages:
        print("No valid adversarial samples to classify after prompt construction. Exiting.")
        return


    print("\nClassifying original samples (Mistral)...")
    original_predictions = classify_samples( # Removed parse_answer from here, it's done inside
        tokenizer, model, original_prompt_messages, batch_size=args.batch_size
    )

    print("\nClassifying adversarial samples (Mistral)...")
    adversarial_predictions = classify_samples( # Removed parse_answer from here
        tokenizer, model, adversarial_prompt_messages, batch_size=args.batch_size
    )
    
    # Ensure predictions lists are not empty and their lengths match valid_samples_count
    if len(original_predictions) != valid_samples_count:
        print(f"Warning: Original predictions count ({len(original_predictions)}) mismatch with valid samples count ({valid_samples_count}).")
    if len(adversarial_predictions) != valid_samples_count:
        print(f"Warning: Adversarial predictions count ({len(adversarial_predictions)}) mismatch with valid samples count ({valid_samples_count}).")


    # Pass the original df, the predictions (which correspond to non-skipped samples),
    # the count of valid samples, and the list of skipped indices.
    df_results = compare_results(
        df.copy(), 
        original_predictions,
        adversarial_predictions,
        valid_samples_count, 
        args.export_flipped_csv,
        skipped_samples_indices=skipped_samples_indices # Pass the skipped indices here
    )
    # df_results.to_csv(args.output_file, index=False)
    # print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main() 