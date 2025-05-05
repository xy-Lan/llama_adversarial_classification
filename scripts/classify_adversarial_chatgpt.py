# scripts/classify_adversarial_chatgpt.py

import pandas as pd
import time
import os
import openai
from openai import OpenAI

# Set up your OpenAI API key
client = OpenAI(
    api_key="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")


def load_data(file_path):
    # Load data with specific range if needed
    # df = pd.read_csv(file_path, skiprows=list(range(1, 101)))
    # For entire dataset:
    df = pd.read_csv(file_path)
    # df = pd.read_csv(file_path, nrows=10)
    return df


def construct_prompts(df):
    original_prompts = []
    adversarial_prompts = []
    skipped_samples = []  # Track skipped samples

    for index, row in df.iterrows():
        original_sample = row['original_samples']
        adversarial_sample = row['adversarial_samples']

        # Check if original sample is a string
        if not isinstance(original_sample, str):
            print(f"Skipping sample at index {index}: Original sample not a string.")
            skipped_samples.append(index)
            continue

        original_sample = original_sample.strip()

        # Check if adversarial sample is a string
        if not isinstance(adversarial_sample, str):
            print(f"Skipping sample at index {index}: Adversarial sample not a string.")
            skipped_samples.append(index)
            continue

        adversarial_sample = adversarial_sample.strip()

        # Check for separator in both samples
        if "~" not in original_sample or "~" not in adversarial_sample:
            print(f"Skipping sample at index {index}: Missing '~' separator in one or both samples.")
            skipped_samples.append(index)
            continue

        # Check if original sample is properly formatted
        original_parts = original_sample.split("~", 1)
        if len(original_parts) != 2 or not original_parts[0].strip() or not original_parts[1].strip():
            print(f"Skipping sample at index {index}: Improperly formatted original sample.")
            skipped_samples.append(index)
            continue

        # Check if adversarial sample is properly formatted
        adversarial_parts = adversarial_sample.split("~", 1)
        if len(adversarial_parts) != 2 or not adversarial_parts[0].strip() or not adversarial_parts[1].strip():
            print(f"Skipping sample at index {index}: Improperly formatted adversarial sample.")
            skipped_samples.append(index)
            continue

        # Build prompts
        evidence_original, claim_original = original_parts
        evidence_adversarial, claim_adversarial = adversarial_parts

        original_prompt = (
            f"Evidence: {evidence_original.strip()}\n"
            f"Claim: {claim_original.strip()}\n"
            "Question: Is this claim supported, refuted, or not enough information based on the evidence?\n"
            "Answer with only one of the following: SUPPORTED, REFUTED, or NOT ENOUGH INFO."
        )
        adversarial_prompt = (
            f"Evidence: {evidence_adversarial.strip()}\n"
            f"Claim: {claim_adversarial.strip()}\n"
            "Question: Is this claim supported, refuted, or not enough information based on the evidence?\n"
            "Answer with only one of the following: SUPPORTED, REFUTED, or NOT ENOUGH INFO."
        )

        original_prompts.append(original_prompt)
        adversarial_prompts.append(adversarial_prompt)

    # Stats on valid samples
    total_samples = len(df)
    valid_samples = total_samples - len(skipped_samples)

    print(f"Total samples: {total_samples}")
    print(f"Skipped samples: {len(skipped_samples)}")
    print(f"Valid samples: {valid_samples}")

    return original_prompts, adversarial_prompts, skipped_samples, valid_samples


def chat_completion(prompt, max_retries=3, retry_delay=5):
    """
    Get completion from ChatGPT with retry logic for rate limits
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="o1",  # Or any other model you prefer
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that classifies claims based on evidence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Keep deterministic
                max_tokens=50
            )
            return response.choices[0].message.content
        except openai.RateLimitError:
            print(f"Rate limit hit. Waiting {retry_delay} seconds before retry.")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(retry_delay)

    return "ERROR"  # Return error if all retries fail


def classify_samples(prompts, batch_size=10, checkpoint_prefix='default'):
    """
    Classify samples with ChatGPT, processing in batches and saving progress
    """
    predictions = []
    checkpoint_file = f'{checkpoint_prefix}_chatgpt_classification.txt'

    # Check if checkpoint exists and load progress
    start_idx = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            saved_predictions = f.read().strip().split('\n')
            if saved_predictions and saved_predictions[0]:
                predictions = saved_predictions
                start_idx = len(predictions)
                print(f"Resuming from checkpoint with {start_idx} predictions")

    for i in range(start_idx, len(prompts)):
        prompt = prompts[i]
        print(f"Processing prompt {i + 1}/{len(prompts)}")

        output_text = chat_completion(prompt)
        prediction = parse_answer(output_text)
        predictions.append(prediction)
        print(f"Prediction for prompt {i + 1}: {prediction}")

        # Save checkpoint every batch_size samples
        if (i + 1) % batch_size == 0 or i == len(prompts) - 1:
            with open(checkpoint_file, 'w') as f:
                f.write('\n'.join(predictions))
            print(f"Checkpoint saved after {i + 1} samples")

        # Add delay to avoid rate limiting
        time.sleep(1)

    return predictions


def parse_answer(output_text):
    """
    Parse the model's response to extract classification
    """
    output_text = output_text.strip().upper()

    if 'SUPPORTED' in output_text:
        return 'SUPPORTED'
    elif 'REFUTED' in output_text:
        return 'REFUTED'
    elif 'NOT ENOUGH INFO' in output_text or 'NOT ENOUGH INFORMATION' in output_text:
        return 'NOT ENOUGH INFO'
    else:
        print(f"Unexpected answer: {output_text}")
        return 'UNKNOWN'


def compare_results(df, original_predictions, adversarial_predictions, valid_samples, model_name="o1"):
    """
    Compare original and adversarial predictions, calculate metrics and export results
    """
    # Add prediction columns
    df['original_prediction'] = original_predictions
    df['adversarial_prediction'] = adversarial_predictions

    # Detect if predictions flipped
    df['prediction_flipped'] = df['original_prediction'] != df['adversarial_prediction']

    # Calculate flip rate
    total_samples = len(df)
    flipped_samples = df['prediction_flipped'].sum()
    flip_rate = flipped_samples / valid_samples if valid_samples > 0 else 0

    # Calculate similarity-weighted flip rate
    df_preserve = df[df['agreed_labels'] == 0]  # Samples that preserve meaning
    flipped_preserve_samples = df_preserve['prediction_flipped'].sum()
    similarity_weighted_flip_rate = flipped_preserve_samples / valid_samples if valid_samples else 0

    # Export flipped samples
    flipped_samples_df = df[df['prediction_flipped'] == True]
    flipped_csv_path = f'./data/{model_name}_flipped_samples.csv'
    flipped_samples_df.to_csv(flipped_csv_path, index=False)
    print(f"Successfully exported {len(flipped_samples_df)} flipped samples to '{flipped_csv_path}'")

    # Export preserved meaning flipped samples
    preserved_flipped_df = df[(df['agreed_labels'] == 0) & (df['prediction_flipped'] == True)]
    preserved_csv_path = f'./data/{model_name}_preserved_flipped_samples.csv'
    preserved_flipped_df.to_csv(preserved_csv_path, index=False)
    print(
        f"Successfully exported {len(preserved_flipped_df)} preserved meaning flipped samples to '{preserved_csv_path}'")

    # Output results
    print(f"Total samples: {total_samples}")
    print(f"Total flipped samples: {flipped_samples}")
    print(f"Flip Rate: {flip_rate:.2%}")
    print(f"Total preserved meaning samples: {len(df_preserve)}")
    print(f"Flipped samples in preserved meaning: {flipped_preserve_samples}")
    print(f"Similarity-Weighted Flip Rate: {similarity_weighted_flip_rate:.2%}")

    return df


def calculate_cost(total_prompts, avg_input_tokens=200, avg_output_tokens=20, model="o1"):
    """
    Calculate the estimated cost of API calls
    """
    # Define price rates for different models
    price_rates = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "o1": {"input": 15.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50}
    }

    if model not in price_rates:
        print(f"Warning: Unknown model '{model}'. Using gpt-4o rates for estimation.")
        model = "gpt-4o"

    # Calculate costs
    input_cost = (total_prompts * avg_input_tokens / 1_000_000) * price_rates[model]["input"]
    output_cost = (total_prompts * avg_output_tokens / 1_000_000) * price_rates[model]["output"]
    total_cost = input_cost + output_cost

    return {
        "input_tokens": total_prompts * avg_input_tokens,
        "output_tokens": total_prompts * avg_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }


def main():
    model_name = "o1"  # Change to the specific model you're using

    # Load dataset
    df = load_data('./data/adversarial_dataset.csv')

    # Construct prompts
    original_prompts, adversarial_prompts, skipped_samples, valid_samples = construct_prompts(df)

    # Calculate estimated cost before starting
    total_prompts = len(original_prompts) + len(adversarial_prompts)
    cost_estimate = calculate_cost(total_prompts, model=model_name)
    print(f"\nEstimated cost for {total_prompts} classifications:")
    print(f"Input tokens: {cost_estimate['input_tokens']:,}")
    print(f"Output tokens: {cost_estimate['output_tokens']:,}")
    print(f"Input cost: ${cost_estimate['input_cost']:.2f}")
    print(f"Output cost: ${cost_estimate['output_cost']:.2f}")
    print(f"Total estimated cost: ${cost_estimate['total_cost']:.2f}\n")

    # Track start time for runtime calculation
    start_time = time.time()

    # Classify original samples with a unique checkpoint file
    print("Classifying original samples...")
    original_predictions = classify_samples(original_prompts, checkpoint_prefix='original')

    # Classify adversarial samples with a different checkpoint file
    print("Classifying adversarial samples...")
    adversarial_predictions = classify_samples(adversarial_prompts, checkpoint_prefix='adversarial')

    # Create valid dataframe by removing skipped samples
    valid_df = df.drop(index=skipped_samples).reset_index(drop=True)

    # Compare results
    result_df = compare_results(valid_df, original_predictions, adversarial_predictions, valid_samples, model_name)

    # Save full results
    result_df.to_csv(f'./data/{model_name}_classification_results.csv', index=False)
    print(f"Full results saved to './data/{model_name}_classification_results.csv'")

    # Calculate and display runtime and final cost
    end_time = time.time()
    runtime_minutes = (end_time - start_time) / 60

    # Calculate actual tokens used (this would need to be tracked during API calls)
    # For now we'll use our estimates
    print(f"\nClassification completed in {runtime_minutes:.2f} minutes")
    print(f"Final cost estimate: ${cost_estimate['total_cost']:.2f}")
    print(f"Average cost per sample: ${cost_estimate['total_cost'] / valid_samples:.4f}")


if __name__ == "__main__":
    main()