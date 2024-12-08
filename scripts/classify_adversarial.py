# scripts/classify_adversarial.py

import pandas as pd
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from accelerate import init_empty_weights, infer_auto_device_map
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df.head(100)


def construct_prompts(df):
    original_prompts = []
    adversarial_prompts = []
    skipped_samples = []  # 记录被剔除的样本索引

    for index, row in df.iterrows():
        original_sample = row['original_samples']
        adversarial_sample = row['adversarial_samples']

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
            print(f"Skipping sample at index {index}: Missing '~' separator in one or both samples.")
            skipped_samples.append(index)
            continue

        # 检查原始样本分割结果是否有效
        original_parts = original_sample.split("~", 1)
        if len(original_parts) != 2 or not original_parts[0].strip() or not original_parts[1].strip():
            print(f"Skipping sample at index {index}: Improperly formatted original sample.")
            skipped_samples.append(index)
            continue

        # 检查对抗性样本分割结果是否有效
        adversarial_parts = adversarial_sample.split("~", 1)
        if len(adversarial_parts) != 2 or not adversarial_parts[0].strip() or not adversarial_parts[1].strip():
            print(f"Skipping sample at index {index}: Improperly formatted adversarial sample.")
            skipped_samples.append(index)
            continue

        # 构建原始和对抗性 Prompts
        evidence_original, claim_original = original_parts
        evidence_adversarial, claim_adversarial = adversarial_parts

        original_prompt = (
            f"Evidence: {evidence_original.strip()}\n"
            f"Claim: {claim_original.strip()}\n"
            "Question: Is this claim supported, refuted, or not enough information based on the evidence?\n"
            "Answer:"
        )
        adversarial_prompt = (
            f"Evidence: {evidence_adversarial.strip()}\n"
            f"Claim: {claim_adversarial.strip()}\n"
            "Question: Is this claim supported, refuted, or not enough information based on the evidence?\n"
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


def load_model(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct", token="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"):
    # tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    # model = LlamaForCausalLM.from_pretrained(model_dir)
    # model.eval()
    print("Loading model...")
    # 配置 INT8 量化
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,  # 启用 INT8 量化
        llm_int8_enable_fp32_cpu_offload=True,  # 在 CPU 上保留 FP32 精度
    )

    # 加载分布式模型（空权重加载）
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            use_auth_token=token,
            device_map="auto"  # 自动分配到可用 GPU 和 CPU
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    # model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
    model.eval()
    print("Model loaded successfully!")
    return tokenizer, model


def classify_samples(tokenizer, model, prompts):
    predictions = []
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i + 1}/{len(prompts)}: {prompt}")  # 增加更多的输出语句，显示当前处理进度
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + 10,
                do_sample=False
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prediction = parse_answer(output_text)
        predictions.append(prediction)
        print(f"Prediction for prompt {i + 1}: {prediction}")  # 增加输出预测结果
    return predictions


def parse_answer(output_text):
    # 提取模型生成的答案部分
    answer = output_text.split('Answer:')[-1].strip().upper()
    if 'SUPPORTED' in answer:
        return 'SUPPORTED'
    elif 'REFUTED' in answer:
        return 'REFUTED'
    elif 'NOT ENOUGH INFO' in answer:
        return 'NOT ENOUGH INFO'
    else:
        print("Answer is ", answer)
        return 'UNKNOWN'


def compare_results(df, original_predictions, adversarial_predictions, valid_samples):
    # 添加预测结果列
    df['original_prediction'] = original_predictions
    df['adversarial_prediction'] = adversarial_predictions

    # 检测分类结果是否翻转
    df['prediction_flipped'] = df['original_prediction'] != df['adversarial_prediction']

    # 计算 Flip Rate
    total_samples = len(df)  # 所有样本的数量
    flipped_samples = df['prediction_flipped'].sum()  # 分类结果翻转的样本数量
    flip_rate = flipped_samples / valid_samples if valid_samples > 0 else 0

    # 计算 Similarity-Weighted Flip Rate
    df_preserve = df[df['agreed_labels'] == 0]  # 保留原义的样本
    flipped_preserve_samples = df_preserve['prediction_flipped'].sum()  # 保留原义中翻转的样本数量
    similarity_weighted_flip_rate = flipped_preserve_samples / valid_samples if valid_samples else 0

    # 输出结果
    print(f"Total samples: {total_samples}")
    print(f"Total flipped samples: {flipped_samples}")
    print(f"Flip Rate: {flip_rate:.2%}")
    print(f"Total preserved meaning samples (agreed_labels == 0): {len(df_preserve)}")
    print(f"Flipped samples in preserved meaning: {flipped_preserve_samples}")
    print(f"Similarity-Weighted Flip Rate: {similarity_weighted_flip_rate:.2%}")

    return df


def main():
    # 加载数据
    df = load_data('./data/adversarial_dataset.csv')

    # 构建提示
    original_prompts, adversarial_prompts, skipped_samples, valid_samples = construct_prompts(df)

    token = "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"

    # 加载模型
    tokenizer, model = load_model(token=token)

    # 对原始样本进行分类
    print("Classifying original samples...")
    original_predictions = classify_samples(tokenizer, model, original_prompts)

    # 对对抗性样本进行分类
    print("Classifying adversarial samples...")
    adversarial_predictions = classify_samples(tokenizer, model, adversarial_prompts)

    valid_df = df.drop(index=skipped_samples).reset_index(drop=True)

    # 比较结果
    result_df = compare_results(valid_df, original_predictions, adversarial_predictions, valid_samples)

    # 保存结果
    # result_df.to_csv('/content/llama_adversarial_classification/scripts/classification_results.csv', index=False)
    # print("Results saved to './data/classification_results.csv'")


if __name__ == "__main__":
    main()



