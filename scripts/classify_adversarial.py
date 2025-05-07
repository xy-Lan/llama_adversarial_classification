# scripts/classify_adversarial.py

import pandas as pd
import torch
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_data(file_path):
    # 与原CPU版本保持一致，只加载前50行数据
    df = pd.read_csv(file_path, nrows=50)
    return df


def construct_prompts(df):
    original_prompts = []
    adversarial_prompts = []
    skipped_samples = []  # 记录被剔除的样本索引

    for index, row in df.iterrows():
        original_sample = row["original_samples"]
        adversarial_sample = row["adversarial_samples"]

        # 检查原始样本是否为字符串
        if not isinstance(original_sample, str):
            print(f"Skipping sample at index {index}: Original sample not a string.")
            skipped_samples.append(index)
            continue

        original_sample = original_sample.strip()

        # 检查对抗性样本是否为字符串
        if not isinstance(adversarial_sample, str):
            print(f"Skipping sample at index {index}: Adversarial sample not a string.")
            skipped_samples.append(index)
            continue

        adversarial_sample = adversarial_sample.strip()

        # 检查分隔符是否存在（针对原始和对抗性样本）
        if "~" not in original_sample or "~" not in adversarial_sample:
            print(
                f"Skipping sample at index {index}: Missing '~' separator in one or both samples."
            )
            skipped_samples.append(index)
            continue

        # 检查原始样本分割结果是否有效
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

        # 检查对抗性样本分割结果是否有效
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

        # 构建原始和对抗性 Prompts - 使用与CPU版本完全相同的格式
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

    # 统计有效样本总数
    total_samples = len(df)
    valid_samples = total_samples - len(skipped_samples)

    print(f"Total samples: {total_samples}")
    print(f"Skipped samples: {len(skipped_samples)}")
    print(f"Valid samples: {valid_samples}")

    return original_prompts, adversarial_prompts, skipped_samples, valid_samples


def load_model(model_name, token=None):
    print("Loading model...")

    # 检查是否有GPU可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name}  {gpu.total_memory/1e9:.1f} GB")
    else:
        print("GPU not available, using CPU for inference.")

    # 加载tokenizer - 保持简单，与CPU版本一致
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

    # 加载模型 - 使用更直接的方式，与CPU版本更接近
    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=token,
        torch_dtype=torch.float32,  # 使用与CPU版本相同的精度
    )

    # 使用标准的PyTorch方式将模型移动到设备
    model = model.to(device)
    model.eval()

    # 验证模型确实在正确的设备上
    device_check = next(model.parameters()).device
    print(f"Model loaded successfully on {device_check}!")

    return tokenizer, model


def classify_samples(tokenizer, model, prompts):
    predictions = []
    device = next(model.parameters()).device

    # 完全保持与CPU版本相同的处理方式，一个一个样本处理
    for i, prompt in enumerate(prompts):
        print(
            f"Processing prompt {i + 1}/{len(prompts)}: {prompt}"
        )  # 与CPU版本完全相同的输出

        # 编码输入 - 使用与CPU版本相同的encode方法
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # 与CPU版本保持相同的生成参数
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids, max_length=input_ids.shape[1] + 10, do_sample=False
            )

        # 解码输出 - 使用与CPU版本相同的decode方法
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prediction = parse_answer(output_text)
        predictions.append(prediction)
        print(f"Prediction for prompt {i + 1}: {prediction}")  # 与CPU版本相同的输出格式

    return predictions


def parse_answer(output_text):
    # 提取模型生成的答案部分 - 与CPU版本完全相同的处理逻辑
    answer = output_text.split("Answer:")[-1].strip().upper()
    if "SUPPORTED" in answer:
        return "SUPPORTED"
    elif "REFUTED" in answer:
        return "REFUTED"
    else:
        print("Answer is ", answer)  # 与CPU版本相同的输出
        return "UNKNOWN"


def compare_results(df, original_predictions, adversarial_predictions, valid_samples):
    # 添加预测结果列
    df["original_prediction"] = original_predictions
    df["adversarial_prediction"] = adversarial_predictions

    # 检测分类结果是否翻转
    df["prediction_flipped"] = df["original_prediction"] != df["adversarial_prediction"]

    # 计算 Flip Rate
    total_samples = len(df)  # 所有样本的数量
    flipped_samples = df["prediction_flipped"].sum()  # 分类结果翻转的样本数量
    flip_rate = flipped_samples / valid_samples if valid_samples > 0 else 0

    # 计算 Similarity-Weighted Flip Rate
    df_preserve = df[df["agreed_labels"] == 0]  # 保留原义的样本
    flipped_preserve_samples = df_preserve[
        "prediction_flipped"
    ].sum()  # 保留原义中翻转的样本数量
    similarity_weighted_flip_rate = (
        flipped_preserve_samples / valid_samples if valid_samples else 0
    )

    # 输出结果
    print(f"Total samples: {total_samples}")
    print(f"Total flipped samples: {flipped_samples}")
    print(f"Flip Rate: {flip_rate:.2%}")
    print(f"Total preserved meaning samples (agreed_labels == 0): {len(df_preserve)}")
    print(f"Flipped samples in preserved meaning: {flipped_preserve_samples}")
    print(f"Similarity-Weighted Flip Rate: {similarity_weighted_flip_rate:.2%}")

    # 显示翻转的例子
    if flipped_samples > 0:
        print("\nFlipped sample examples:")
        flipped_indices = df.index[df["prediction_flipped"]].tolist()[
            : min(3, flipped_samples)
        ]
        for idx in flipped_indices:
            print(f"Sample {idx}:")
            print(f"  Original: {df.loc[idx, 'original_samples']}")
            print(f"  Adversarial: {df.loc[idx, 'adversarial_samples']}")
            print(f"  Original Prediction: {df.loc[idx, 'original_prediction']}")
            print(f"  Adversarial Prediction: {df.loc[idx, 'adversarial_prediction']}")
            print()

    return df


def main():
    # 命令行参数
    parser = argparse.ArgumentParser(
        description="Classify FEVER dataset with Llama model"
    )
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.2-1B-Instruct", help="Model name or path"
    )
    parser.add_argument("--token", help="Hugging Face token")
    parser.add_argument(
        "--data_path",
        default="./data/adversarial_dataset_corrected.csv",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output_dir", default="./results", help="Directory to save results"
    )
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    df = load_data(args.data_path)

    # 构建提示
    original_prompts, adversarial_prompts, skipped_samples, valid_samples = (
        construct_prompts(df)
    )

    # 加载模型
    tokenizer, model = load_model(args.model, args.token)

    # 对原始样本进行分类
    print("\nClassifying original samples...")
    original_predictions = classify_samples(tokenizer, model, original_prompts)

    # 对对抗性样本进行分类
    print("\nClassifying adversarial samples...")
    adversarial_predictions = classify_samples(tokenizer, model, adversarial_prompts)

    # 移除被跳过的样本
    valid_df = df.drop(index=skipped_samples).reset_index(drop=True)

    # 比较结果
    result_df = compare_results(
        valid_df, original_predictions, adversarial_predictions, valid_samples
    )

    # 保存结果
    result_path = os.path.join(
        args.output_dir, f"{args.model.split('/')[-1]}_results.csv"
    )
    result_df.to_csv(result_path, index=False)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
