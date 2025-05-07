# scripts/classify_adversarial.py

import pandas as pd
import torch
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import time


def load_data(file_path, nrows=None):
    """加载数据，支持加载部分或全部数据"""
    if nrows is not None:
        df = pd.read_csv(file_path, nrows=nrows)
        print(f"Loaded {nrows} rows from {file_path}")
    else:
        df = pd.read_csv(file_path)
        print(f"Loaded all {len(df)} rows from {file_path}")
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
    """加载模型和分词器，支持多种加载选项"""
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
    }

    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    try:
        print(f"Loading model with config: {model_kwargs}")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to basic loading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, use_auth_token=token, torch_dtype=torch.float32
        )
        if torch.cuda.is_available():
            model = model.to("cuda")

    # 设置评估模式
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded successfully on {device}")

    return tokenizer, model


def classify_samples(tokenizer, model, prompts, batch_size=8):
    predictions = []
    device = next(model.parameters()).device

    # 如果有GPU，调整批处理大小
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch_size = min(batch_size, max(1, int(mem) // 2))
        print(f"Using batch size: {batch_size}")

    # 批量处理
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        print(
            f"Processing batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}"
        )

        try:
            # 使用padding处理批次
            inputs = tokenizer(batch, padding=True, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,  # 只生成少量token
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # 处理每个输出
            for j, output in enumerate(outputs):
                # 获取输入长度以找到生成的部分
                input_length = inputs.input_ids[j].size(0)
                generated_tokens = output[input_length:]

                # 解码生成的文本
                generated_text = tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                full_text = batch[j] + generated_text

                # 解析答案
                prediction = parse_answer(full_text)
                predictions.append(prediction)

                # 打印详情
                print(f"Prediction for sample {i+j+1}: {prediction}")

                # 打印前几个样本的详细信息
                if i == 0 and j < 3:
                    print(f"  Sample {j+1}:")
                    print(f"    Prompt: {batch[j]}")
                    print(f"    Generated: {generated_text}")
                    print(f"    Full text: {full_text}")

        except Exception as e:
            print(f"Error in batch processing: {e}")
            # 单条回退
            for j, prompt in enumerate(batch):
                try:
                    print(f"Processing individual prompt {i+j+1}")
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        output = model.generate(
                            **inputs, max_new_tokens=10, do_sample=False
                        )

                    # 解码生成的文本
                    generated_text = tokenizer.decode(
                        output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    full_text = prompt + generated_text

                    # 解析答案
                    prediction = parse_answer(full_text)
                    predictions.append(prediction)
                    print(f"Prediction for sample {i+j+1}: {prediction}")

                except Exception as inner_e:
                    print(f"Error in single sample processing: {inner_e}")
                    predictions.append("UNKNOWN")

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
    """解析模型输出，提取SUPPORTED或REFUTED标签"""
    # 提取模型生成的答案部分
    answer_part = output_text.split("Answer:")[-1].strip().upper()

    # 简化的模式匹配，只要包含相关关键词即可
    if "SUPPORT" in answer_part and "NOT SUPPORT" not in answer_part:
        return "SUPPORTED"
    elif "REFUTE" in answer_part or "NOT SUPPORT" in answer_part:
        return "REFUTED"
    # 基本上所有生成的文本都应该包含上述关键词之一
    else:
        # 后备策略：偏向确定结果而非未知
        print(f"Unrecognized answer, using fallback strategy: '{answer_part}'")
        if any(word in answer_part for word in ["YES", "TRUE", "CORRECT"]):
            return "SUPPORTED"
        elif any(word in answer_part for word in ["NO", "FALSE", "INCORRECT", "NOT"]):
            return "REFUTED"
        # 实在无法判断的情况
        else:
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
    parser = argparse.ArgumentParser(description="分类FEVER数据集中的对抗样本")
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.2-1B-Instruct", help="模型名称或路径"
    )
    parser.add_argument("--token", help="Hugging Face访问令牌")
    parser.add_argument(
        "--data_path",
        default="./data/adversarial_dataset_corrected.csv",
        help="数据集路径",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="批处理大小 (默认: 8)"
    )
    parser.add_argument("--output_dir", default="./results", help="结果输出目录")
    parser.add_argument(
        "--nrows", type=int, default=50, help="加载的行数 (默认: 50，设为0加载全部)"
    )
    parser.add_argument(
        "--save_predictions", action="store_true", help="保存每个样本的预测结果"
    )
    parser.add_argument("--verbose", action="store_true", help="输出详细日志")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志级别
    if args.verbose:
        print("启用详细日志模式")

    # 加载数据
    df = load_data(args.data_path, args.nrows if args.nrows > 0 else None)

    # 构建提示
    start_time = time.time()
    original_prompts, adversarial_prompts, skipped_samples, valid_samples = (
        construct_prompts(df)
    )
    prep_time = time.time() - start_time
    print(f"提示构建完成，耗时 {prep_time:.2f} 秒")

    # 加载模型
    tokenizer, model = load_model(args.model, args.token)

    # 对原始样本进行分类
    print("\n分类原始样本...")
    orig_start = time.time()
    original_predictions = classify_samples(
        tokenizer, model, original_prompts, args.batch_size
    )
    orig_time = time.time() - orig_start
    print(f"原始样本分类完成，耗时 {orig_time:.2f} 秒")

    # 对对抗性样本进行分类
    print("\n分类对抗性样本...")
    adv_start = time.time()
    adversarial_predictions = classify_samples(
        tokenizer, model, adversarial_prompts, args.batch_size
    )
    adv_time = time.time() - adv_start
    print(f"对抗样本分类完成，耗时 {adv_time:.2f} 秒")

    # 移除被跳过的样本
    valid_df = df.drop(index=skipped_samples).reset_index(drop=True)

    # 比较结果
    result_df = compare_results(
        valid_df, original_predictions, adversarial_predictions, valid_samples
    )

    # 输出性能统计
    total_time = time.time() - start_time
    samples_per_sec = valid_samples / (orig_time + adv_time)
    print(f"\n性能统计:")
    print(f"  总处理时间: {total_time:.2f} 秒")
    print(f"  样本处理速度: {samples_per_sec:.2f} 样本/秒")

    # 保存结果
    model_name_short = args.model.split("/")[-1]
    batch_info = f"_batch{args.batch_size}"
    sample_info = f"_n{args.nrows}" if args.nrows > 0 else "_full"
    result_path = os.path.join(
        args.output_dir, f"{model_name_short}{batch_info}{sample_info}_results.csv"
    )
    result_df.to_csv(result_path, index=False)
    print(f"结果已保存到 {result_path}")

    # 可选：保存每个样本的预测结果
    if args.save_predictions:
        pred_path = os.path.join(
            args.output_dir,
            f"{model_name_short}{batch_info}{sample_info}_predictions.csv",
        )
        pred_df = pd.DataFrame(
            {
                "original_prompt": original_prompts,
                "adversarial_prompt": adversarial_prompts,
                "original_prediction": original_predictions,
                "adversarial_prediction": adversarial_predictions,
                "flipped": [
                    o != a
                    for o, a in zip(original_predictions, adversarial_predictions)
                ],
            }
        )
        pred_df.to_csv(pred_path, index=False)
        print(f"预测详情已保存到 {pred_path}")

    return result_df


if __name__ == "__main__":
    main()
