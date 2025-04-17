import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


def load_data(file_path):
    """加载测试数据集"""
    df = pd.read_csv(file_path)
    return df


def construct_prompts(df):
    """构建测试提示"""
    original_prompts = []
    adversarial_prompts = []
    skipped_samples = []

    for index, row in df.iterrows():
        try:
            original_sample = row['original_samples']
            adversarial_sample = row['adversarial_samples']

            # 检查样本有效性
            if not isinstance(original_sample, str) or not isinstance(adversarial_sample, str):
                print(f"跳过样本 {index}: 样本非字符串")
                skipped_samples.append(index)
                continue

            # 分割样本
            evidence_original, claim_original = original_sample.split('~', 1)
            evidence_adversarial, claim_adversarial = adversarial_sample.split('~', 1)

            # 构建提示
            original_prompt = (
                f"You are a fact verification assistant. Given evidence and a claim, "
                f"determine if the claim is SUPPORTED, REFUTED, or if there's NOT ENOUGH INFO "
                f"based solely on the provided evidence.\n\n"
                f"Evidence: {evidence_original.strip()}\n"
                f"Claim: {claim_original.strip()}\n"
                f"Question: Is this claim supported, refuted, or not enough information based on the evidence?\n"
                f"Answer:"
            )

            adversarial_prompt = (
                f"You are a fact verification assistant. Given evidence and a claim, "
                f"determine if the claim is SUPPORTED, REFUTED, or if there's NOT ENOUGH INFO "
                f"based solely on the provided evidence.\n\n"
                f"Evidence: {evidence_adversarial.strip()}\n"
                f"Claim: {claim_adversarial.strip()}\n"
                f"Question: Is this claim supported, refuted, or not enough information based on the evidence?\n"
                f"Answer:"
            )

            original_prompts.append(original_prompt)
            adversarial_prompts.append(adversarial_prompt)

        except Exception as e:
            print(f"处理样本 {index} 时出错: {e}")
            skipped_samples.append(index)

    # 统计样本信息
    total_samples = len(df)
    valid_samples = total_samples - len(skipped_samples)

    print(f"总样本数: {total_samples}")
    print(f"跳过样本数: {len(skipped_samples)}")
    print(f"有效样本数: {valid_samples}")

    return original_prompts, adversarial_prompts, skipped_samples, valid_samples


def load_model(base_model_name, peft_model_path):
    """加载微调后的PEFT模型"""
    print("加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print("加载PEFT适配器...")
    model = PeftModel.from_pretrained(base_model, peft_model_path)

    # 合并适配器
    model = model.merge_and_unload()
    model.eval()

    print("模型加载成功!")
    return tokenizer, model


def classify_samples(tokenizer, model, prompts):
    """对样本进行分类"""
    predictions = []
    for i, prompt in enumerate(prompts):
        print(f"处理提示 {i + 1}/{len(prompts)}")
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + 20,
                do_sample=False
            )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prediction = parse_answer(output_text)
        predictions.append(prediction)
        print(f"提示 {i + 1} 的预测结果: {prediction}")

    return predictions


def parse_answer(output_text):
    """解析模型生成的答案"""
    answer = output_text.split('Answer:')[-1].strip().upper()

    if 'SUPPORTED' in answer:
        return 'SUPPORTED'
    elif 'REFUTED' in answer:
        return 'REFUTED'
    elif 'NOT ENOUGH INFO' in answer:
        return 'NOT ENOUGH INFO'
    else:
        print("未知答案: ", answer)
        return 'UNKNOWN'


def compare_results(df, original_predictions, adversarial_predictions, valid_samples):
    """比较分类结果"""
    # 添加预测结果列
    df['original_prediction'] = original_predictions
    df['adversarial_prediction'] = adversarial_predictions

    # 检测分类结果是否翻转
    df['prediction_flipped'] = df['original_prediction'] != df['adversarial_prediction']

    # 计算 Flip Rate
    total_samples = len(df)
    flipped_samples = df['prediction_flipped'].sum()
    flip_rate = flipped_samples / valid_samples if valid_samples > 0 else 0

    # 计算 Similarity-Weighted Flip Rate
    df_preserve = df[df['agreed_labels'] == 0]  # 保留原义的样本
    flipped_preserve_samples = df_preserve['prediction_flipped'].sum()
    similarity_weighted_flip_rate = flipped_preserve_samples / valid_samples if valid_samples else 0

    # 导出翻转样本
    # flipped_samples_df = df[df['prediction_flipped'] == True]
    # flipped_samples_df.to_csv('./data/finetuned_peft_model_flipped_samples.csv', index=False)
    # print(f"成功导出 {len(flipped_samples_df)} 个翻转样本到 './data/finetuned_peft_model_flipped_samples.csv'")
    #
    # # 导出保留原义且翻转的样本
    # preserved_flipped_df = df[(df['agreed_labels'] == 0) & (df['prediction_flipped'] == True)]
    # preserved_flipped_df.to_csv('./data/finetuned_peft_model_preserved_flipped_samples.csv', index=False)
    # print(
    #     f"成功导出 {len(preserved_flipped_df)} 个保留原义的翻转样本到 './data/finetuned_peft_model_preserved_flipped_samples.csv'")

    # 输出结果
    print(f"总样本数: {total_samples}")
    print(f"总翻转样本数: {flipped_samples}")
    print(f"翻转率: {flip_rate:.2%}")
    print(f"保留原义样本总数: {len(df_preserve)}")
    print(f"保留原义样本中的翻转数: {flipped_preserve_samples}")
    print(f"相似性加权翻转率: {similarity_weighted_flip_rate:.2%}")

    return df


def main():
    # 基础模型名称
    base_model_name = "meta-llama/Llama-3.2-1B-Instruct"

    # 加载测试数据
    df = load_data('./data/final_test_set.csv')

    # 构建提示
    original_prompts, adversarial_prompts, skipped_samples, valid_samples = construct_prompts(df)

    # 加载微调后的PEFT模型
    peft_model_path = "models/llama-1b-adversarial-finetuned"
    tokenizer, model = load_model(base_model_name, peft_model_path)

    # 对原始样本进行分类
    print("分类原始样本...")
    original_predictions = classify_samples(tokenizer, model, original_prompts)

    # 对对抗性样本进行分类
    print("分类对抗性样本...")
    adversarial_predictions = classify_samples(tokenizer, model, adversarial_prompts)

    # 删除跳过的样本
    valid_df = df.drop(index=skipped_samples).reset_index(drop=True)

    # 比较结果
    result_df = compare_results(valid_df, original_predictions, adversarial_predictions, valid_samples)

    # 保存完整结果
    # result_df.to_csv('./data/finetuned_peft_model_classification_results.csv', index=False)
    # print("结果已保存到 './data/finetuned_peft_model_classification_results.csv'")


if __name__ == "__main__":
    main()