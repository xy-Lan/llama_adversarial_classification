# scripts/classify_adversarial.py

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import os


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def construct_prompts(df):
    original_prompts = []
    adversarial_prompts = []
    skipped_samples = []

    for index, row in df.iterrows():
        original_sample = row['original_samples']
        adversarial_sample = row['adversarial_samples']

        if not isinstance(original_sample, str) or not isinstance(adversarial_sample, str):
            skipped_samples.append(index)
            continue

        original_sample = original_sample.strip()
        adversarial_sample = adversarial_sample.strip()

        if "~" not in original_sample or "~" not in adversarial_sample:
            skipped_samples.append(index)
            continue

        original_parts = original_sample.split("~", 1)
        adversarial_parts = adversarial_sample.split("~", 1)

        if len(original_parts) != 2 or len(adversarial_parts) != 2:
            skipped_samples.append(index)
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

    valid_samples = len(df) - len(skipped_samples)
    print(f"Total samples: {len(df)} | Skipped: {len(skipped_samples)} | Valid: {valid_samples}")

    return original_prompts, adversarial_prompts, skipped_samples, valid_samples


def load_model(model_name="meta-llama/Llama-3.2-1B-Instruct", token=None):
    print("Loading model...")

    # 检查GPU是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 如果是H100，配置进一步优化
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_properties.name}")
        # 为H100进行特定优化
        if "H100" in gpu_properties.name:
            print("H100 GPU detected! Optimizing for H100...")
            # 对于H100，启用TF32可以获得更好的性能
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # 配置8位量化或BF16混合精度以减少内存使用并加速
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,  # 启用8位量化
        llm_int8_enable_fp32_cpu_offload=True,  # 需要时卸载到CPU
        llm_int8_threshold=6.0,  # 量化阈值
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

    # 使用设备映射自动处理模型加载到GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=token,
        quantization_config=quant_config,  # 使用量化
        device_map="auto",  # 自动管理模型在GPU和CPU之间的分配
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,  # 使用BF16（如果支持）
    )

    model.eval()
    print("Model loaded successfully!")
    return tokenizer, model


def classify_with_llama(tokenizer, model, prompts, batch_size=32):
    predictions = []

    # 确定设备
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")

    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("设置pad_token = eos_token")

    # 检测是否为H100并自动调整批处理大小
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(0)
        gpu_memory_gb = gpu_properties.total_memory / 1e9
        print(f"GPU: {gpu_properties.name} with {gpu_memory_gb:.1f} GB memory")

        # 如果是H100，自动调整批处理大小
        if "H100" in gpu_properties.name:
            # H100 通常有80GB内存，可以处理更大批次
            batch_size = min(batch_size, 128)  # 上限为128，但用户可以通过参数覆盖
            print(f"H100 detected! Using batch size: {batch_size}")
        else:
            # 基于GPU内存估算合适的批处理大小
            estimated_batch_size = int(gpu_memory_gb * 2)  # 每GB内存估计处理2个样本
            batch_size = min(batch_size, estimated_batch_size)
            print(f"Using adjusted batch size: {batch_size} based on available GPU memory")

    # 批处理预测
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        print(
            f"Processing batch {i // batch_size + 1}/{(len(prompts) - 1) // batch_size + 1}: samples {i + 1}-{min(i + batch_size, len(prompts))}")

        try:
            # 对批次进行编码
            batch_inputs = tokenizer(batch_prompts, padding=True, return_tensors='pt')

            # 将输入移至与模型相同的设备
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

            with torch.no_grad():
                # 使用批处理生成
                outputs = model.generate(
                    **batch_inputs,
                    max_length=batch_inputs['input_ids'].shape[1] + 10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # 处理每个输出
            for j, output_ids in enumerate(outputs):
                output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                prediction = parse_answer(output_text)
                predictions.append(prediction)
                if (i + j + 1) % 10 == 0:  # 每10个样本输出一次，减少日志量
                    print(f"Processed {i + j + 1}/{len(prompts)} samples")

        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            print("Falling back to individual processing for this batch")

            # 如果批处理失败，尝试逐个处理
            for j, prompt in enumerate(batch_prompts):
                try:
                    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
                    with torch.no_grad():
                        output_ids = model.generate(
                            input_ids=input_ids,
                            max_length=input_ids.shape[1] + 10,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    prediction = parse_answer(output_text)
                    predictions.append(prediction)
                    print(f"Processed individual sample {i + j + 1}/{len(prompts)}: {prediction}")
                except Exception as inner_e:
                    print(f"Error processing individual sample {i + j + 1}: {inner_e}")
                    predictions.append("UNKNOWN")  # 添加默认预测

    return predictions


def parse_answer(output_text):
    answer = output_text.split('Answer:')[-1].strip().upper()
    if 'SUPPORTED' in answer:
        return 'SUPPORTED'
    elif 'REFUTED' in answer:
        return 'REFUTED'
    else:
        return 'UNKNOWN'


def compare_results_with_accuracy(df, original_preds, adversarial_preds, valid_samples, model_name,
                                  output_dir="./results"):
    # 添加预测结果列
    df['original_prediction'] = original_preds
    df['adversarial_prediction'] = adversarial_preds
    df['prediction_flipped'] = df['original_prediction'] != df['adversarial_prediction']

    # 计算正确性（原始样本预测是否与标准答案一致）
    df['correct'] = df['original_prediction'] == df['correctness'].str.upper()

    # 计算Clean Accuracy
    clean_accuracy = df['correct'].sum() / len(df) if len(df) > 0 else 0

    # 计算翻转率
    total_flipped = df['prediction_flipped'].sum()
    flip_rate = total_flipped / valid_samples if valid_samples > 0 else 0

    # 计算在判断正确的样本中的翻转率
    df_correct = df[df['correct'] == True]
    flipped_correct = df_correct['prediction_flipped'].sum()
    correct_flip_rate = flipped_correct / len(df_correct) if len(df_correct) > 0 else 0

    # 计算在保留原义且判断正确的样本中的翻转率
    df_preserve_and_correct = df[(df['agreed_labels'] == 0) & (df['correct'] == True)]
    flipped_preserve_correct = df_preserve_and_correct['prediction_flipped'].sum()
    preserve_correct_flip_rate = flipped_preserve_correct / len(df_preserve_and_correct) if len(
        df_preserve_and_correct) > 0 else 0

    # 创建结果目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取简短的模型名称
    model_short_name = model_name.split('/')[-1]

    # 创建TXT文件路径
    txt_output_path = os.path.join(output_dir, f"{model_short_name}_results.txt")

    # 将结果写入TXT文件
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        f.write(f"===== {model_short_name} 分类结果 =====\n\n")
        f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("基本统计:\n")
        f.write(f"总样本数: {len(df)}\n")
        f.write(f"有效样本数: {valid_samples}\n")

        f.write("\n精度评估:\n")
        f.write(f"Clean Accuracy: {clean_accuracy:.2%} ({df['correct'].sum()}/{len(df)})\n")

        f.write("\n翻转率统计:\n")
        f.write(f"所有样本的翻转率: {flip_rate:.2%} ({total_flipped}/{valid_samples})\n")

        f.write("\n正确预测样本分析:\n")
        f.write(f"正确预测的样本数: {len(df_correct)}\n")
        f.write(f"正确预测样本中的翻转率: {correct_flip_rate:.2%} ({flipped_correct}/{len(df_correct)})\n")

        f.write("\n保留原义且正确预测样本分析:\n")
        f.write(f"保留原义且正确预测的样本数: {len(df_preserve_and_correct)}\n")
        f.write(
            f"保留原义且正确预测样本中的翻转率: {preserve_correct_flip_rate:.2%} ({flipped_preserve_correct}/{len(df_preserve_and_correct)})\n")

        f.write("\n详细分类结果:\n")
        f.write(f"原样本SUPPORTED预测: {sum(1 for p in original_preds if p == 'SUPPORTED')}\n")
        f.write(f"原样本REFUTED预测: {sum(1 for p in original_preds if p == 'REFUTED')}\n")
        f.write(f"原样本UNKNOWN预测: {sum(1 for p in original_preds if p == 'UNKNOWN')}\n")
        f.write(f"对抗样本SUPPORTED预测: {sum(1 for p in adversarial_preds if p == 'SUPPORTED')}\n")
        f.write(f"对抗样本REFUTED预测: {sum(1 for p in adversarial_preds if p == 'REFUTED')}\n")
        f.write(f"对抗样本UNKNOWN预测: {sum(1 for p in adversarial_preds if p == 'UNKNOWN')}\n")

    print(f"\n结果已保存到: {txt_output_path}")

    # 输出结果
    print(f"\n===== 分类结果 =====")
    print(f"Total samples: {len(df)}")
    print(f"Clean Accuracy: {clean_accuracy:.2%} ({df['correct'].sum()}/{len(df)})")
    print(f"\nAll samples flip rate: {flip_rate:.2%} ({total_flipped}/{valid_samples})")
    print(f"\nCorrect predictions: {len(df_correct)}")
    print(f"Flip rate among correct predictions: {correct_flip_rate:.2%} ({flipped_correct}/{len(df_correct)})")
    print(f"\nPreserve meaning + correct predictions: {len(df_preserve_and_correct)}")
    print(
        f"Flip rate among preserve meaning + correct: {preserve_correct_flip_rate:.2%} ({flipped_preserve_correct}/{len(df_preserve_and_correct)})")

    return df


def export_incorrect_predictions(df, model_name, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)
    model_short_name = model_name.split('/')[-1]
    output_path = os.path.join(output_dir, f"{model_short_name}_misclassified_samples.csv")

    incorrect_df = df[df['original_prediction'] != df['correctness'].str.upper()][[
        'original_samples', 'adversarial_samples', 'original_prediction', 'correctness'
    ]]
    incorrect_df.to_csv(output_path, index=False)
    print(f"Exported {len(incorrect_df)} misclassified samples to {output_path}")


def main():
    # 导入必要的库
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import pandas as pd
    import time
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LLama分类器评估对抗性示例')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='模型名称')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--batch_size', type=int, default=8, help='批处理大小')
    parser.add_argument('--data_path', type=str, default='./data/adversarial_dataset_corrected.csv', help='数据集路径')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出结果目录')
    args = parser.parse_args()

    # 打印GPU信息
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
        print(f"CUDA device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    start_time = time.time()

    # 加载数据
    print(f"Loading data from {args.data_path}")
    df = load_data(args.data_path)

    # 构建提示
    original_prompts, adversarial_prompts, skipped_samples, valid_samples = construct_prompts(df)

    # 加载模型
    tokenizer, model = load_model(model_name=args.model, token=args.token)

    # 设置CUDA事件以测量时间
    if torch.cuda.is_available():
        torch_start = torch.cuda.Event(enable_timing=True)
        torch_end = torch.cuda.Event(enable_timing=True)
        torch_start.record()

    # 对原始样本进行分类
    print("\nClassifying original samples...")
    original_predictions = classify_with_llama(tokenizer, model, original_prompts)

    # 对对抗性样本进行分类
    print("\nClassifying adversarial samples...")
    adversarial_predictions = classify_with_llama(tokenizer, model, adversarial_prompts)

    # 计算分类时间
    if torch.cuda.is_available():
        torch_end.record()
        torch.cuda.synchronize()
        print(f"\nClassification completed in {torch_start.elapsed_time(torch_end) / 1000:.2f} seconds")

    # 处理有效的数据帧
    valid_df = df.drop(index=skipped_samples).reset_index(drop=True)

    # 比较结果并输出TXT文件
    result_df = compare_results_with_accuracy(valid_df, original_predictions, adversarial_predictions,
                                              valid_samples, args.model, args.output_dir)

    # 导出错误分类的样本
    export_incorrect_predictions(result_df, args.model, args.output_dir)

    # 保存完整结果
    model_short_name = args.model.split('/')[-1]
    result_output_path = os.path.join(args.output_dir, f"{model_short_name}_classification_results.csv")
    result_df.to_csv(result_output_path, index=False)
    print(f"Full results saved to '{result_output_path}'")

    # 计算总运行时间
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()



