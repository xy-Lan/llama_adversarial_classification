import pandas as pd
import torch
from openai import OpenAI
import os
import time
import numpy as np

# 设置你的 OpenAI API 密钥
client = OpenAI(
    api_key="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"
)


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def construct_prompts(df):
    original_prompts = []
    adversarial_prompts = []
    skipped_samples_indices = []

    for index, row in df.iterrows():
        original_sample = row["original_samples"]
        adversarial_sample = row["adversarial_samples"]

        if not isinstance(original_sample, str) or not isinstance(
            adversarial_sample, str
        ):
            skipped_samples_indices.append(index)
            continue

        original_sample = original_sample.strip()
        adversarial_sample = adversarial_sample.strip()

        if "~" not in original_sample or "~" not in adversarial_sample:
            skipped_samples_indices.append(index)
            continue

        original_parts = original_sample.split("~", 1)
        adversarial_parts = adversarial_sample.split("~", 1)

        if len(original_parts) != 2 or len(adversarial_parts) != 2:
            skipped_samples_indices.append(index)
            continue

        evidence_original, claim_original = original_parts
        evidence_adversarial, claim_adversarial = adversarial_parts

        original_prompt = (
            f"Evidence: {evidence_original.strip()}\n"
            f"Claim: {claim_original.strip()}\n"
            "Question: Is this claim supported or refuted based on the evidence?\n"
            "Answer:"
        )
        adversarial_prompt = (
            f"Evidence: {evidence_adversarial.strip()}\n"
            f"Claim: {claim_adversarial.strip()}\n"
            "Question: Is this claim supported or refuted based on the evidence?\n"
            "Answer:"
        )

        original_prompts.append(original_prompt)
        adversarial_prompts.append(adversarial_prompt)

    valid_samples_count = len(df) - len(skipped_samples_indices)
    print(
        f"Total samples: {len(df)} | Skipped: {len(skipped_samples_indices)} | Valid: {valid_samples_count}"
    )

    return original_prompts, adversarial_prompts, skipped_samples_indices, valid_samples_count


def classify_with_gpt(prompts, model="gpt-4"):
    predictions = []
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i + 1}/{len(prompts)} with {model}")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            answer = response.choices[0].message.content
            prediction = parse_answer(answer)
        except Exception as e:
            print(f"Error on sample {i} with model {model}: {e}")
            prediction = "ERROR"
        predictions.append(prediction)
        print(f"Model {model} Prediction: {prediction}")
    return predictions


def parse_answer(output_text):
    if not isinstance(output_text, str):
        return "UNKNOWN"
    answer = output_text.split("Answer:")[-1].strip().upper()
    if "SUPPORTED" in answer:
        return "SUPPORTED"
    elif "REFUTED" in answer:
        return "REFUTED"
    else:
        print(f"Warning: Could not parse GPT answer: '{output_text}'. Marking as UNKNOWN.")
        return "UNKNOWN"


def compare_results(df, original_predictions, adversarial_predictions, valid_samples, export_flipped_csv=None):
    original_pred_series = pd.Series([pd.NA] * len(df), index=df.index)
    adversarial_pred_series = pd.Series([pd.NA] * len(df), index=df.index)

    skipped_indices = df.attrs.get("skipped_samples_indices", [])
    processed_indices = df.index.difference(skipped_indices).tolist()

    if len(processed_indices) == len(original_predictions) and len(
        processed_indices
    ) == len(adversarial_predictions):
        df.loc[processed_indices, "original_prediction"] = original_predictions
        df.loc[processed_indices, "adversarial_prediction"] = adversarial_predictions
    else:
        print(
            f"Warning: Mismatch in length between processed indices ({len(processed_indices)}) and predictions (orig: {len(original_predictions)}, adv: {len(adversarial_predictions)}). Predictions may not be aligned correctly."
        )
        if len(df) == len(original_predictions) and len(df) == len(adversarial_predictions) and not skipped_indices:
             df["original_prediction"] = original_predictions
             df["adversarial_prediction"] = adversarial_predictions

    df["comparison_result"] = "Not Run"

    if "correctness" not in df.columns:
        print(
            "\nWarning: 'correctness' column not found. Skipping Clean Accuracy and related metrics."
        )
        clean_accuracy = 0
        df["original_is_correct"] = False
        num_correct_original_samples_valid_gt = 0
        num_samples_with_valid_gt = 0
    else:
        df["mapped_correctness"] = (
            df["correctness"]
            .astype(str)
            .str.upper()
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

        df["original_is_correct"] = (
            (df["original_prediction"].astype(str) == df["mapped_correctness"].astype(str))
             & df["is_valid_ground_truth"]
        )
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

    df["prediction_flipped"] = df["original_prediction"].astype(str) != df["adversarial_prediction"].astype(str)
    valid_prediction_mask = df["original_prediction"].notna() & df["adversarial_prediction"].notna() & \
                            ~df["original_prediction"].astype(str).isin(["ERROR", "UNKNOWN"]) & \
                            ~df["adversarial_prediction"].astype(str).isin(["ERROR", "UNKNOWN"])
    
    df["prediction_flipped"] = (df["original_prediction"].astype(str) != df["adversarial_prediction"].astype(str)) & valid_prediction_mask

    num_comparable_predictions = valid_prediction_mask.sum()

    flipped_samples_count = df["prediction_flipped"].sum()
    
    overall_flip_rate = (
        flipped_samples_count / num_comparable_predictions if num_comparable_predictions > 0 else 0
    )

    if "agreed_labels" in df.columns and pd.api.types.is_numeric_dtype(df["agreed_labels"]):
        meaning_preserving_df = df[(df["agreed_labels"] == 0) & valid_prediction_mask].copy()
    else:
        print("\nWarning: 'agreed_labels' column not found or not numeric. Skipping meaning-preserving metrics.")
        meaning_preserving_df = pd.DataFrame(columns=df.columns)
        
    num_meaning_preserving_comparable = len(meaning_preserving_df)
    flips_in_meaning_preserving = meaning_preserving_df["prediction_flipped"].sum()

    flip_rate_mp_vs_all_comparable = (
        flips_in_meaning_preserving / num_comparable_predictions if num_comparable_predictions > 0 else 0
    )

    if "correctness" in df.columns and "original_is_correct" in df.columns and not meaning_preserving_df.empty:
        correct_and_meaning_preserving_df = meaning_preserving_df[
            meaning_preserving_df["original_is_correct"]
        ].copy()
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
            f"Meaning-Preserving & Correctly Classified Original Samples (Comparable): {num_correct_and_mp}"
        )
        print(
            f"Flip Rate (for Meaning-Preserving & Correctly Classified Originals): {robust_flip_rate:.2%} ({flips_in_correct_and_mp}/{num_correct_and_mp})"
        )
        
        if num_correct_and_mp > 0:
            targeted_mp_asr = flips_in_correct_and_mp / num_correct_and_mp
        else:
            targeted_mp_asr = float("nan")
        if num_correct_and_mp > 0:
            original_targeted_mp_flip_rate = flips_in_meaning_preserving / num_correct_and_mp
        else:
            original_targeted_mp_flip_rate = float("nan")
        print(
            f"Original Targeted MP Flip Rate (All MP Flips / (MP & Correct Originals)): {original_targeted_mp_flip_rate:.2%} ({flips_in_meaning_preserving}/{num_correct_and_mp})"
        )

    else:
        print(
            "Robust Flip Rate metrics: Not calculated ('correctness' column missing or no meaning-preserving samples)."
        )
        num_correct_and_mp = 0
        flips_in_correct_and_mp = 0
        robust_flip_rate = 0

    print(f"\nOverall Metrics (based on {num_comparable_predictions} samples with comparable predictions):")
    print(f"  Total Flipped Samples: {flipped_samples_count}")
    print(f"  Overall Flip Rate: {overall_flip_rate:.2%}")

    print(f"\nMeaning-Preserving Metrics (Total comparable MP samples: {num_meaning_preserving_comparable} samples):")
    print(f"  Flipped Samples within Meaning-Preserving (Comparable): {flips_in_meaning_preserving}")
    print(
        f"  Flip Rate (Meaning-Preserving Flips / All Comparable Predictions): {flip_rate_mp_vs_all_comparable:.2%}"
    )
    
    flip_rate_within_mp_comparable = flips_in_meaning_preserving / num_meaning_preserving_comparable if num_meaning_preserving_comparable > 0 else 0
    print(f"  Flip Rate (within Meaning-Preserving comparable subset): {flip_rate_within_mp_comparable:.2%}")

    if export_flipped_csv:
        os.makedirs(os.path.dirname(export_flipped_csv), exist_ok=True)
        
        if "correctness" in df.columns and "original_is_correct" in df.columns and "agreed_labels" in df.columns and pd.api.types.is_numeric_dtype(df["agreed_labels"]):
            base_df_for_numerator = df[
                (df["agreed_labels"] == 0) &
                df["original_is_correct"] &
                valid_prediction_mask
            ].copy()
            
            flipped_samples_to_export_df = base_df_for_numerator[
                base_df_for_numerator["prediction_flipped"] == True
            ].copy()

            if not flipped_samples_to_export_df.empty:
                columns_to_select = [
                    "original_samples", 
                    "adversarial_samples", 
                    "correctness", 
                    "agreed_labels", 
                    "original_prediction", 
                    "adversarial_prediction",
                    "prediction_flipped"
                ]
                final_columns_to_export = [col for col in columns_to_select if col in df.columns]
                
                export_df = df.loc[flipped_samples_to_export_df.index, final_columns_to_export].copy()
                
                export_df.to_csv(export_flipped_csv, index=False)
                print(f"\nExported {len(export_df)} samples (numerator of 'Robust Flip Rate') to: {export_flipped_csv}")
            else:
                print(f"\nNo samples to export for 'Robust Flip Rate' numerator (path: {export_flipped_csv}).")
        else:
            print(f"\nCould not export 'Robust Flip Rate' numerator samples to {export_flipped_csv} due to missing required columns or no such samples.")

    return df


def main():
    output_dir = "./results/gpt_classification"
    os.makedirs(output_dir, exist_ok=True)

    input_csv_file = "./data/adversarial_dataset_corrected.csv"
    gpt_model_name = "gpt-4.1"

    print(f"Loading data from: {input_csv_file}")
    df_full = load_data(input_csv_file)

    print("Constructing prompts...")
    original_prompts, adversarial_prompts, skipped_samples_indices, valid_samples_count = (
        construct_prompts(df_full.copy())
    )
    
    df_full.attrs["skipped_samples_indices"] = skipped_samples_indices

    print(f"Classifying original samples with {gpt_model_name}...")
    original_preds = classify_with_gpt(original_prompts, model=gpt_model_name)
    
    print(f"Classifying adversarial samples with {gpt_model_name}...")
    adversarial_preds = classify_with_gpt(adversarial_prompts, model=gpt_model_name)

    print("\nComparing results...")
    flipped_export_path = os.path.join(output_dir, f"{gpt_model_name.replace('/', '_')}_robust_flipped_samples.csv")

    result_df = compare_results(
        df_full, 
        original_preds, 
        adversarial_preds, 
        valid_samples_count,
        export_flipped_csv=flipped_export_path
    )

    full_results_csv_path = os.path.join(output_dir, f"{gpt_model_name.replace('/', '_')}_full_classification_results.csv")
    result_df.to_csv(full_results_csv_path, index=False)
    print(f"\nFull classification results saved to: {full_results_csv_path}")

    print(f"\n===== GPT Classification for {gpt_model_name} Finished =====")
    print(f"Detailed console output above. CSV results in: {output_dir}")


if __name__ == "__main__":
    # if not os.environ.get("OPENAI_API_KEY"):
    #     print("Warning: OPENAI_API_KEY environment variable not set. API calls might fail.")
    main()