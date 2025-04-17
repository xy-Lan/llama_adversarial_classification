import os
import logging
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import warnings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


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

    # 检查列是否存在prediction列
    has_predictions = 'original_prediction' in df.columns and 'adversarial_prediction' in df.columns

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

            # 获取语义一致性标签
            semantics_preserved = row['agreed_labels'] == 0.0

            # 获取输出标签
            if not has_predictions or pd.isna(row.get('original_prediction')):
                output_original = "SUPPORTED"
            else:
                output_original = row['original_prediction']

            # 创建指令
            instruction = (
                "You are a fact verification assistant. Given evidence and a claim, "
                "determine if the claim is SUPPORTED, REFUTED, or if there's NOT ENOUGH INFO "
                "based solely on the provided evidence."
            )

            # 添加原始样本
            formatted_data.append({
                "instruction": instruction,
                "input": f"Evidence: {evidence_original.strip()}\nClaim: {claim_original.strip()}\nQuestion: Is this claim supported, refuted, or not enough information based on the evidence?",
                "output": output_original
            })

            # 仅为保留语义的样本添加对抗性样本
            if semantics_preserved:
                formatted_data.append({
                    "instruction": instruction,
                    "input": f"Evidence: {evidence_adversarial.strip()}\nClaim: {claim_adversarial.strip()}\nQuestion: Is this claim supported, refuted, or not enough information based on the evidence?",
                    "output": output_original
                })

                # 添加一致性示例
                consistency_instruction = (
                    "You are a fact verification assistant. Given two different presentations "
                    "of the same evidence and claim with identical meaning, you should provide "
                    "consistent verdicts regardless of phrasing differences."
                )

                consistency_input = (
                    f"Presentation 1:\nEvidence: {evidence_original.strip()}\nClaim: {claim_original.strip()}\n\n"
                    f"Presentation 2:\nEvidence: {evidence_adversarial.strip()}\nClaim: {claim_adversarial.strip()}\n\n"
                    f"Question: Do these presentations warrant the same verdict (SUPPORTED, REFUTED, or NOT ENOUGH INFO)?"
                )

                consistency_output = "Yes, these presentations have identical meaning and warrant the same verdict."

                formatted_data.append({
                    "instruction": consistency_instruction,
                    "input": consistency_input,
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
        output_dir="models/llama-1b-adversarial-finetuned",
        epochs=3,
        learning_rate=2e-4,
        batch_size=8,
        gradient_accumulation_steps=4,
        use_8bit=True,
        use_4bit=False,
        lora_rank=16,
        lora_alpha=32,
        max_seq_length=512,
        huggingface_token=None
):
    """主微调函数"""
    # 配置警告过滤
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

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

    # 设置量化配置
    quantization_config = None
    if use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=True
        )
    elif use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

    # 选择设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载分词器
    logging.info(f"加载模型和分词器: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=huggingface_token,
            trust_remote_code=True
        )
    except Exception as e:
        logging.error(f"加载分词器失败: {e}")
        return None

    # 确保分词器具有必要的特殊标记
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            use_auth_token=huggingface_token,
            trust_remote_code=True,
            device_map={"": device}
        )
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        return None

    # 设置Gradient Checkpointing以节省内存
    model.gradient_checkpointing_enable()

    # 准备模型进行训练
    if use_8bit or use_4bit:
        model = prepare_model_for_kbit_training(model)

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
    try:
        trainer.train()
    except Exception as e:
        logging.error(f"训练过程中出错: {e}")
        return None

    # 保存最终模型
    logging.info("保存模型...")
    trainer.save_model()

    logging.info(f"训练完成！模型已保存到 {output_dir}")
    return trainer


def main():
    """主程序入口"""
    # 设置Hugging Face token(如果需要访问Llama模型)
    hf_token = os.environ.get('HF_TOKEN', None)  # 建议使用环境变量

    # 如果没有设置环境变量，可以在这里直接设置token
    # hf_token = "your_huggingface_token"

    # 检查token
    if not hf_token:
        logging.warning("未提供Hugging Face Token，可能无法访问私有模型")

    # 执行微调
    finetune_llama(
        training_data_file="data/final_complete_training_set.csv",
        output_dir="models/llama-1b-adversarial-finetuned",
        huggingface_token=hf_token,
        epochs=3,  # 根据需要调整
        batch_size=8,  # 根据GPU内存调整
        use_8bit=True  # 使用8位量化以节省内存
    )


if __name__ == "__main__":
    main()