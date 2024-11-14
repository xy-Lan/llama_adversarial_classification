# scripts/classify_adversarial.py

import pandas as pd
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def construct_prompts(df):
    original_prompts = []
    adversarial_prompts = []
    for index, row in df.iterrows():
        original_sample = row['original_samples']
        adversarial_sample = row['adversarial_samples']

        # 构建新的提示，要求模型判断句子的真实性
        original_prompt = f"Statement: {original_sample}\nQuestion: Is this statement true or false based on common knowledge? Answer with 'SUPPORTED', 'REFUTED', or 'NOT ENOUGH INFO'.\nAnswer:"
        adversarial_prompt = f"Statement: {adversarial_sample}\nQuestion: Is this statement true or false based on common knowledge? Answer with 'SUPPORTED', 'REFUTED', or 'NOT ENOUGH INFO'.\nAnswer:"

        original_prompts.append(original_prompt)
        adversarial_prompts.append(adversarial_prompt)
    return original_prompts, adversarial_prompts


def load_model(model_name="meta-llama/Llama-3.2-1B"):
    # tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    # model = LlamaForCausalLM.from_pretrained(model_dir)
    # model.eval()
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    print("Model loaded successfully!")
    return tokenizer, model


def classify_samples(tokenizer, model, prompts):
    predictions = []
    for prompt in prompts:
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
        return 'UNKNOWN'


def compare_results(df, original_predictions, adversarial_predictions):
    df['original_prediction'] = original_predictions
    df['adversarial_prediction'] = adversarial_predictions

    # 检测分类结果是否翻转
    df['prediction_flipped'] = df['original_prediction'] != df['adversarial_prediction']

    # 只考虑人工标注为保留原意的样本（agreed_labels == 0）
    df_preserve = df[df['agreed_labels'] == 0]

    # 在保留原意的情况下，统计分类结果翻转的比例
    total_preserve_samples = len(df_preserve)
    flipped_preserve_samples = df_preserve['prediction_flipped'].sum()
    flip_rate_preserve = flipped_preserve_samples / total_preserve_samples if total_preserve_samples > 0 else 0

    print(f"Total samples where meaning is preserved (agreed_labels == 0): {total_preserve_samples}")
    print(f"Classification flipped on {flipped_preserve_samples} samples where meaning is preserved.")
    print(f"Flip rate in preserved meaning samples: {flip_rate_preserve:.2%}")

    return df


def main():
    # 加载数据
    df = load_data('./data/adversarial_dataset.csv')

    # 构建提示
    original_prompts, adversarial_prompts = construct_prompts(df)

    # 加载模型
    tokenizer, model = load_model()

    # 对原始样本进行分类
    print("Classifying original samples...")
    original_predictions = classify_samples(tokenizer, model, original_prompts)

    # 对对抗性样本进行分类
    print("Classifying adversarial samples...")
    adversarial_predictions = classify_samples(tokenizer, model, adversarial_prompts)

    # 比较结果
    result_df = compare_results(df, original_predictions, adversarial_predictions)

    # 保存结果
    result_df.to_csv('./data/classification_results.csv', index=False)
    print("Results saved to './data/classification_results.csv'")


if __name__ == "__main__":
    main()



