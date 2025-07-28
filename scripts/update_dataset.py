import pandas as pd
import argparse

def process_csv(input_path, output_path, binary_semantic=False):
    df = pd.read_csv(input_path)
    rows = []

    for i, row in df.iterrows():
        # Optional: binary mapping
        if binary_semantic:
            semantic = 1 if row["agreed_labels"] in [1, 2] else 0
        else:
            semantic = row["agreed_labels"]

        # Add adversarial sample
        rows.append({
            "text": row["adversarial_samples"],
            "pair_id": i,
            "is_adv": 1,
            "semantic": semantic
        })

        # Add original sample
        rows.append({
            "text": row["original_samples"],
            "pair_id": i,
            "is_adv": 0,
            "semantic": semantic
        })

    processed_df = pd.DataFrame(rows)
    processed_df.to_csv(output_path, index=False)
    print(f"✅ Saved processed dataset to: {output_path}")
    print(processed_df.head(4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to original train.csv")
    parser.add_argument("--output", required=True, help="Path to save processed CSV")
    parser.add_argument("--binary", action="store_true", help="Convert semantic to binary (1 = changed or nonsense)")
    args = parser.parse_args()

    process_csv(args.input, args.output, args.binary)