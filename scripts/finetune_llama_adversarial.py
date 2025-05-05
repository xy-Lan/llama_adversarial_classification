#!/usr/bin/env python3
import os
import logging
import warnings
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 忽略警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def setup_device_info():
    """打印系统设备信息"""
    logging.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"当前CUDA设备: {torch.cuda.current_device()}")
        logging.info(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def prepare_training_data(csv_file):
    """从CSV加载数据并转换为微调格式"""
    logging.info(f"加载训练数据: {csv_file}")

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logging.error(f"读取CSV文件时出错: {e}")
        return []

    formatted_data = []
    skipped_rows = 0

    for idx, row in df.iterrows():
        try:
            # 处理原始样本
            original_sample = row['original_samples']
            if not isinstance(original_sample, str) or '~' not in original_sample:
                logging.warning(f"跳过行 {idx}: 原始样本格式不正确")
                skipped_rows += 1
                continue

            evidence_original, claim_original = original_sample.split('~', 1)

            # 处理对抗性样本
            adversarial_sample = row['adversarial_samples']
            if not isinstance(adversarial_sample, str) or '~' not in adversarial_sample:
                logging.warning(f"跳过行 {idx}: 对抗性样本格式不正确")
                skipped_rows += 1
                continue

            evidence_adversarial, claim_adversarial = adversarial_sample.split('~', 1)

            # 根据 agreed_labels 决定语义一致性
            if row['agreed_labels'] == 0:  # 保留原义
                # 语义保留的情况
                consistency_output = "CONSISTENT"

                # 第一个训练样本：原始样本
                formatted_data.append({
                    "instruction": "你是一个语义一致性检查助手。判断两个不同表述的声明是否保留原始含义。",
                    "input": f"Evidence: {evidence_original.strip()}\nClaim: {claim_original.strip()}",
                    "output": "ORIGINAL"
                })

                # 第二个训练样本：对抗性样本
                formatted_data.append({
                    "instruction": "你是一个语义一致性检查助手。判断两个不同表述的声明是否保留原始含义。",
                    "input": f"Evidence: {evidence_adversarial.strip()}\nClaim: {claim_adversarial.strip()}",
                    "output": "ORIGINAL"
                })

                # 一致性比较样本
                formatted_data.append({
                    "instruction": "你是一个语义一致性检查助手。判断两个不同表述的声明是否保留原始含义。",
                    "input": (
                        f"Presentation 1:\nEvidence: {evidence_original.strip()}\n"
                        f"Claim: {claim_original.strip()}\n\n"
                        f"Presentation 2:\nEvidence: {evidence_adversarial.strip()}\n"
                        f"Claim: {claim_adversarial.strip()}\n\n"
                        f"Question: Do these presentations have the same semantic meaning?"
                    ),
                    "output": consistency_output
                })

            elif row['agreed_labels'] == 1:  # 改变原义
                # 语义改变的情况
                consistency_output = "INCONSISTENT"

                # 不同语义的比较样本
                formatted_data.append({
                    "instruction": "你是一个语义一致性检查助手。判断两个不同表述的声明是否保留原始含义。",
                    "input": (
                        f"Presentation 1:\nEvidence: {evidence_original.strip()}\n"
                        f"Claim: {claim_original.strip()}\n\n"
                        f"Presentation 2:\nEvidence: {evidence_adversarial.strip()}\n"
                        f"Claim: {claim_adversarial.strip()}\n\n"
                        f"Question: Do these presentations have the same semantic meaning?"
                    ),
                    "output": consistency_output
                })

        except Exception as e:
            logging.error(f"处理行 {idx} 时出错: {str(e)}")
            skipped_rows += 1

    logging.info(f"总行数: {len(df)}, 跳过的行数: {skipped_rows}, 处理的行数: {len(df) - skipped_rows}")
    logging.info(f"准备了 {len(formatted_data)} 条训练样本")
    return formatted_data

def finetune_llama(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        training_data_file="data/final_complete_training_set.csv",
        output_dir="models/llama-1b-semantic-consistency",
        epochs=3,
        learning_rate=2e-4,
        batch_size=4,
        gradient_accumulation_steps=8,
        lora_rank=16,
        lora_alpha=32,
        max_seq_length=512,
        huggingface_token=None
):
    """主微调函数"""
    # 打印设备信息
    setup_device_info()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载和准备数据
    training_data = prepare_training_data(training_data_file)
    if len(training_data) == 0:
        logging.error("没有生成训练样本，请检查CSV文件格式和标签")
        return None

    # 创建数据集
    dataset = Dataset.from_dict({
        'instruction': [item['instruction'] for item in training_data],
        'input': [item['input'] for item in training_data],
        'output': [item['output'] for item in training_data]
    })

    # 加载分词器
    logging.info(f"加载模型和分词器: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=huggingface_token,
        trust_remote_code=True
    )

    # 确保分词器具有必要的特殊标记
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_auth_token=huggingface_token,
        trust_remote_code=True,
        device_map="auto"
    )

    # 设置Gradient Checkpointing以节省内存
    model.gradient_checkpointing_enable()

    # 设置LoRA配置
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力模块
        "gate_proj", "up_proj", "down_proj"  # MLP模块
    ]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )

    # 获取PEFT模型
    model = get_peft_model(model, peft_config)

    # 打印可训练参数数量
    model.print_trainable_parameters()

    # 预处理函数
    def preprocess_function(examples):
        # 构建完整提示
        prompts = []
        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            prompt = f"<|begin_of_text|><|user|>\n{instruction}\n\n{input_text}<|end_of_user|>\n\n<|assistant|>\n{output}<|end_of_text|>"
            prompts.append(prompt)

        # 编码文本
        tokenized_examples = tokenizer(
            prompts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length"
        )

        # 创建标签
        tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()

        # 找到助手回复的起始位置
        for i, input_ids in enumerate(tokenized_examples["input_ids"]):
            # 找到assistant标记的位置
            assistant_pos = None
            for pos in range(len(input_ids) - 1):
                if tokenizer.decode(input_ids[pos:pos + 2]).startswith("<|assistant|>"):
                    assistant_pos = pos + 2  # 跳过<|assistant|>标记
                    break

            # 不计算之前的损失
            if assistant_pos:
                tokenized_examples["labels"][i][:assistant_pos] = [-100] * assistant_pos

        return tokenized_examples

    # 预处理数据
    logging.info("预处理数据...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="预处理数据集"
    )

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    # 设置数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        max_length=max_seq_length,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    # 设置训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 开始训练
    logging.info("开始微调...")
    trainer.train()

    # 保存最终模型
    logging.info("保存模型...")
    trainer.save_model()

    logging.info(f"训练完成！模型已保存到 {output_dir}")
    return trainer

def main():
    finetune_llama(
        training_data_file="data/final_complete_training_set.csv",
        output_dir="models/llama-1b-semantic-consistency",
        epochs=3,
        batch_size=4,
        gradient_accumulation_steps=8
    )

if __name__ == "__main__":
    main()











































