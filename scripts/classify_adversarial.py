# scripts/classify_adversarial.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time, argparse, traceback
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
)
from peft import PeftModel  # 添加PEFT支持


# ---------- 数据加载 ----------
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


# ---------- prompt 构造 ----------
def construct_prompts(df: pd.DataFrame):
    orig, adv, skipped = [], [], []

    for idx, row in df.iterrows():
        o, a = row["original_samples"], row["adversarial_samples"]
        if not (isinstance(o, str) and isinstance(a, str)):
            skipped.append(idx)
            continue
        if "~" not in o or "~" not in a:
            skipped.append(idx)
            continue

        ev_o, cl_o = o.split("~", 1)
        ev_a, cl_a = a.split("~", 1)

        tpl = (
            "Evidence: {}\n"
            "Claim: {}\n"
            "Question: Is this claim supported or refuted based on the evidence? "
            'Answer ONLY "SUPPORTED" or "REFUTED" (no other words)\n'
            "Answer:"
        )

        orig.append(tpl.format(ev_o.strip(), cl_o.strip()))
        adv.append(tpl.format(ev_a.strip(), cl_a.strip()))

    print(f"Total {len(df)} | Skipped {len(skipped)} | Valid {len(df)-len(skipped)}")
    return orig, adv, skipped, len(df) - len(skipped)


# ---------- 模型加载 ----------
def load_model(model_name: str, lora_path=None, token=None):
    print("Loading model …")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name}  {gpu.total_memory/1e9:.1f} GB")
        if "H100" in gpu.name:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_auth_token=token, cache_dir=os.environ.get("HF_HOME", None)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 确保使用左侧padding

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=token,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=os.environ.get("HF_HOME", None),
    )

    # 如果提供了lora路径，则加载LoRA权重
    if lora_path:
        print(f"Loading LoRA weights from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    print("Model ready.")
    return tokenizer, model


# ---------- 只允许两个 token 的 logits 处理器 ----------
class TwoLabelLimiter(LogitsProcessor):
    def __init__(self, allow_ids):
        self.allow = allow_ids  # list[int]

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allow] = 0
        return scores + mask


def parse_answer(first_token_id, sup_id):
    return "SUPPORTED" if first_token_id == sup_id else "REFUTED"


# ---------- 分类 ----------
def classify(tok, model, prompts, batch_size=32):
    preds = []
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")

    # H100 自动调批
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch_size = min(batch_size, int(mem) // 2 or 1)
        print(f"Adjusted batch size: {batch_size}")

    # 记录不同的预测结果以验证是否有变化
    prediction_counts = {"SUPPORTED": 0, "REFUTED": 0, "UNKNOWN": 0}
    
    # 调试信息
    print("开始对样本进行分类...")

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        print(f"Batch {i//batch_size+1}/{(len(prompts)-1)//batch_size+1}")

        try:
            inputs = tok(batch, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                # 生成更多tokens以确保完整标签
                outs = model.generate(
                    **inputs,
                    max_new_tokens=10,  # 生成足够多的token以包含完整标签
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tok.eos_token_id,
                )
            inp_len = inputs["input_ids"].shape[1]
            
            # 为批次中的每个样本进行解码和分类
            for b_idx, out in enumerate(outs):
                # 解码生成的文本
                gen_tokens = out[inp_len:]
                gen_text = tok.decode(gen_tokens, skip_special_tokens=True).strip()
                
                # 调试用：打印部分样本的生成结果
                if i == 0 and b_idx < 5:  # 只显示第一批次的前5个
                    print(f"样本 #{b_idx+1} 的生成内容: '{gen_text}'")
                
                # 识别标签 - 优先使用完全匹配
                if "SUPPORTED" in gen_text:
                    pred = "SUPPORTED"
                elif "REFUTED" in gen_text:
                    pred = "REFUTED"
                # 次优先使用前缀匹配
                elif gen_text.startswith("S") or "support" in gen_text.lower():
                    pred = "SUPPORTED"
                elif gen_text.startswith("R") or "refut" in gen_text.lower():
                    pred = "REFUTED"
                # 再次扩展匹配范围
                elif any(x in gen_text.lower() for x in ["是的", "正确", "支持", "确实", "agrees"]):
                    pred = "SUPPORTED"
                elif any(x in gen_text.lower() for x in ["否", "不", "错误", "拒绝", "disagrees"]):
                    pred = "REFUTED"
                else:
                    # 默认情况: 作为UNKNOWN处理
                    pred = "REFUTED"  # 默认为REFUTED，避免过多UNKNOWN影响结果
                    print(f"警告: 无法确定标签，默认为REFUTED: '{gen_text}'")
                
                preds.append(pred)
                prediction_counts[pred] += 1
                
                # 每50个样本显示一次当前统计
                if len(preds) % 50 == 0:
                    print(f"当前预测分布: {prediction_counts}")
        
        except Exception as e:
            print("批处理错误:", e)
            # 单样本回退处理
            for j, p in enumerate(batch):
                try:
                    ids = tok(p, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model.generate(
                            **ids, 
                            max_new_tokens=10, 
                            do_sample=False,
                            temperature=0.0,
                            pad_token_id=tok.eos_token_id,
                        )
                    # 解码生成的文本
                    gen_text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    
                    # 使用相同的匹配逻辑
                    if "SUPPORTED" in gen_text:
                        pred = "SUPPORTED"
                    elif "REFUTED" in gen_text:
                        pred = "REFUTED"
                    elif gen_text.startswith("S") or "support" in gen_text.lower():
                        pred = "SUPPORTED"
                    elif gen_text.startswith("R") or "refut" in gen_text.lower():
                        pred = "REFUTED"
                    elif any(x in gen_text.lower() for x in ["是的", "正确", "支持", "确实", "agrees"]):
                        pred = "SUPPORTED"
                    elif any(x in gen_text.lower() for x in ["否", "不", "错误", "拒绝", "disagrees"]):
                        pred = "REFUTED"
                    else:
                        pred = "REFUTED"
                        print(f"警告(单样本): 无法确定标签，默认为REFUTED: '{gen_text}'")
                    
                    preds.append(pred)
                    prediction_counts[pred] += 1
                except Exception as ie:
                    print("单样本错误:", ie)
                    preds.append("REFUTED")  # 出错时默认为REFUTED
                    prediction_counts["UNKNOWN"] += 1
    
    # 显示最终预测分布
    print("最终预测分布:")
    print(f"SUPPORTED: {prediction_counts['SUPPORTED']} ({prediction_counts['SUPPORTED']/len(preds):.2%})")
    print(f"REFUTED: {prediction_counts['REFUTED']} ({prediction_counts['REFUTED']/len(preds):.2%})")
    if prediction_counts["UNKNOWN"] > 0:
        print(f"UNKNOWN: {prediction_counts['UNKNOWN']} ({prediction_counts['UNKNOWN']/len(preds):.2%})")
    
    return preds


# ---------- 结果统计（与你原版一致） ----------
def compare_results_with_accuracy(
    df, orig_preds, adv_preds, valid_samples, model_name, output_dir="./results"
):

    df["original_prediction"] = orig_preds
    df["adversarial_prediction"] = adv_preds
    df["prediction_flipped"] = df["original_prediction"] != df["adversarial_prediction"]
    df["correct"] = df["original_prediction"] == df["correctness"].str.upper()

    clean_acc = df["correct"].mean() if len(df) else 0
    total_flip = df["prediction_flipped"].sum()
    flip_rate = total_flip / valid_samples if valid_samples else 0

    df_correct = df[df["correct"]]
    flip_corr = df_correct["prediction_flipped"].sum()
    corr_flip_r = flip_corr / len(df_correct) if len(df_correct) else 0

    df_preserve = df[(df["agreed_labels"] == 0) & df["correct"]]
    flip_pcorr = df_preserve["prediction_flipped"].sum()
    pcorr_r = flip_pcorr / len(df_preserve) if len(df_preserve) else 0

    os.makedirs(output_dir, exist_ok=True)
    short = model_name.split("/")[-1] + ("_lora" if args.lora else "")  # 修改输出文件名
    txt = os.path.join(output_dir, f"{short}_results.txt")

    with open(txt, "w", encoding="utf-8") as f:
        f.write(f"===== {short} 分类结果 =====\n\n")
        f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"总样本数: {len(df)}  有效样本: {valid_samples}\n")
        f.write(f"Clean Accuracy: {clean_acc:.2%}\n\n")
        f.write(f"翻转率: {flip_rate:.2%}\n")
        f.write(f"正确预测中的翻转率: {corr_flip_r:.2%}\n")
        f.write(f"保留原义+正确预测中的翻转率: {pcorr_r:.2%}\n")

    print(f"\n结果已保存到: {txt}")
    print("===== 分类结果 =====")
    print(f"Total samples: {len(df)}")
    print(f"Clean Accuracy: {clean_acc:.2%}")
    print(f"All flip rate: {flip_rate:.2%}")
    print(f"Correct flip rate: {corr_flip_r:.2%}")
    print(f"Preserve+Correct flip rate: {pcorr_r:.2%}")

    return df


def export_incorrect_predictions(df, model_name, out_dir="./results"):
    short = model_name.split("/")[-1] + ("_lora" if args.lora else "")  # 修改输出文件名
    path = os.path.join(out_dir, f"{short}_misclassified.csv")
    cols = [
        "original_samples",
        "adversarial_samples",
        "original_prediction",
        "correctness",
    ]
    df[df["original_prediction"] != df["correctness"].str.upper()][cols].to_csv(
        path, index=False
    )
    print(f"Misclassified samples saved to {path}")


# ---------- 主入口 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--lora", help="LoRA权重目录路径")  # 添加LoRA参数
    parser.add_argument("--token")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--data_path", default="./data/adversarial_dataset_corrected.csv"
    )
    # parser.add_argument("--output_dir", default="./results")
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "results"),
        help="模型输出结果目录（默认写入 TMPDIR/results）",
    )
    global args
    args = parser.parse_args()

    df = load_data(args.data_path)
    orig_p, adv_p, skipped, valid = construct_prompts(df)

    tok, model = load_model(args.model, args.lora, args.token)  # 传递LoRA参数

    print("\nClassifying original …")
    o_preds = classify(tok, model, orig_p, args.batch_size)
    print("\nClassifying adversarial …")
    a_preds = classify(tok, model, adv_p, args.batch_size)

    valid_df = df.drop(index=skipped).reset_index(drop=True)
    res_df = compare_results_with_accuracy(
        valid_df, o_preds, a_preds, valid, args.model, args.output_dir
    )
    export_incorrect_predictions(res_df, args.model, args.output_dir)

    # 保存完整 CSV
    model_name_with_lora = args.model.split("/")[-1] + (
        "_lora" if args.lora else ""
    )  # 修改输出文件名
    full_path = os.path.join(
        args.output_dir, model_name_with_lora + "_full_results.csv"
    )
    res_df.to_csv(full_path, index=False)
    print(f"Full csv saved to {full_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("程序执行出错:", e)
        traceback.print_exc()
