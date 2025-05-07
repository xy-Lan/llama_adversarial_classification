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

    # 设置模型加载参数
    model_kwargs = {
        "use_auth_token": token,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
        "cache_dir": os.environ.get("HF_HOME", None),
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_auth_token=token, cache_dir=os.environ.get("HF_HOME", None)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 在HPC环境中，有时可能需要增加超时时间
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as e:
        print(f"Model loading error: {e}")
        print("Trying with extended timeout...")
        model_kwargs["local_files_only"] = False
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # 如果提供了lora路径，则加载LoRA权重
    if lora_path:
        print(f"Loading LoRA weights from {lora_path}")
        try:
            model = PeftModel.from_pretrained(model, lora_path)
            print("LoRA weights loaded successfully!")
        except Exception as e:
            print(f"Error loading LoRA weights: {e}")
            print(traceback.format_exc())
            print("Continuing with base model...")

    model.eval()
    print("Model ready.")
    return tokenizer, model


# ---------- 只允许两个 token 的 logits 处理器 ----------
class TwoLabelLimiter(LogitsProcessor):
    def __init__(self, allow_ids):
        self.allow = allow_ids  # list[int]

    def __call__(self, input_ids, scores):
        # 保存原始分数用于调试
        self.original_scores = scores.clone().detach()

        # 创建掩码，只允许SUPPORTED和REFUTED的token
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allow] = 0

        # 返回调整后的分数
        return scores + mask


# 改进的token寻找函数
def find_label_tokens(tokenizer, labels=["SUPPORTED", "REFUTED"]):
    """更精确地查找标签对应的tokens"""
    token_ids = []
    token_debug_info = []

    for label in labels:
        # 方法1: 直接转换（适用于标签是单个token的情况）
        direct_id = tokenizer.convert_tokens_to_ids(label)

        # 方法2: 使用tokenizer编码（可能分成多个token）
        encoded = tokenizer.encode(label, add_special_tokens=False)
        encoded_tokens = [tokenizer.decode([t]) for t in encoded]

        # 方法3: 编码带空格的版本（某些tokenizer可能会有不同）
        space_encoded = tokenizer.encode(" " + label, add_special_tokens=False)
        space_tokens = [tokenizer.decode([t]) for t in space_encoded]

        # 选择最合适的token ID
        chosen_id = None
        chosen_method = ""

        # 优先选直接匹配的token
        if direct_id != tokenizer.unk_token_id:
            chosen_id = direct_id
            chosen_method = "direct match"
        # 如果是单个token的编码，则选择它
        elif len(encoded) == 1:
            chosen_id = encoded[0]
            chosen_method = "encoded (no space)"
        # 如果带空格版本是单个token，则选择它
        elif len(space_encoded) == 1:
            chosen_id = space_encoded[0]
            chosen_method = "encoded (with space)"
        # 如果都是多个token，则选择第一个
        elif len(encoded) > 0:
            chosen_id = encoded[0]
            chosen_method = "first token of multi-token encoding"
        else:
            # 最后的后备方案
            chosen_id = tokenizer.encode(label, add_special_tokens=False)[0]
            chosen_method = "fallback encoding"

        token_ids.append(chosen_id)

        # 收集调试信息
        debug = {
            "label": label,
            "chosen_id": chosen_id,
            "method": chosen_method,
            "direct_id": direct_id,
            "encoded": encoded,
            "encoded_tokens": encoded_tokens,
            "space_encoded": space_encoded,
            "space_tokens": space_tokens,
            "decoded_back": tokenizer.decode([chosen_id]),
        }
        token_debug_info.append(debug)

    return token_ids, token_debug_info


def parse_answer(first_token_id, sup_id):
    return "SUPPORTED" if first_token_id == sup_id else "REFUTED"


# ---------- 分类 ----------
def classify(tok, model, prompts, batch_size=32, debug_prefix=""):
    preds = []
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")

    # H100 自动调批
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch_size = min(batch_size, int(mem) // 2 or 1)
        print(f"Adjusted batch size: {batch_size}")

    # 使用改进的标签token查找方法
    allow, token_debug = find_label_tokens(tok)

    # 打印详细的Token调试信息
    print("\n===== Detailed Label Token Information =====")
    for debug in token_debug:
        print(f"Label: {debug['label']}")
        print(f"  Chosen token ID: {debug['chosen_id']} via {debug['method']}")
        print(f"  Decodes back to: '{debug['decoded_back']}'")
        print(f"  Direct conversion ID: {debug['direct_id']}")
        print(
            f"  Encoded without space: {debug['encoded']} → {debug['encoded_tokens']}"
        )
        print(
            f"  Encoded with space: {debug['space_encoded']} → {debug['space_tokens']}"
        )
        print("")

    # 简要打印标签信息
    print(f"[DEBUG] SUPPORTED token ID: {allow[0]}, REFUTED token ID: {allow[1]}")
    print(f"[DEBUG] SUPPORTED decodes back to: '{tok.decode([allow[0]])}'")
    print(f"[DEBUG] REFUTED decodes back to: '{tok.decode([allow[1]])}'")

    # 创建logits处理器
    limiter = LogitsProcessorList([TwoLabelLimiter(allow)])

    # 跟踪是否所有预测都是同一个标签
    all_same_prediction = True
    first_prediction = None

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        print(f"Batch {i//batch_size+1}/{(len(prompts)-1)//batch_size+1}")

        try:
            inputs = tok(batch, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                # 为第一个批次添加返回分数
                return_dict_in_generate = i == 0
                outs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tok.eos_token_id,
                    logits_processor=limiter,
                    return_dict_in_generate=return_dict_in_generate,
                    output_scores=return_dict_in_generate,
                )
            inp_len = inputs["input_ids"].shape[1]

            # 如果是第一个批次且返回了分数，打印详细的分数信息
            if i == 0 and return_dict_in_generate and hasattr(outs, "scores"):
                print("\n===== First Batch Token Probability Analysis =====")
                # 获取原始logits
                original_scores = limiter[0].original_scores

                for idx in range(min(3, len(batch))):
                    print(f"\nSample {idx+1}:")
                    print(f"  Prompt: {batch[idx][:100]}...")

                    # 打印两个候选标签的原始logits和softmax概率
                    logits = original_scores[idx]
                    probs = torch.nn.functional.softmax(logits, dim=-1)

                    print("  Raw logits for label tokens:")
                    print(f"    SUPPORTED ({allow[0]}): {logits[allow[0]].item():.4f}")
                    print(f"    REFUTED ({allow[1]}): {logits[allow[1]].item():.4f}")
                    print(
                        f"    Difference: {(logits[allow[0]] - logits[allow[1]]).item():.4f}"
                    )

                    print("  Probabilities for label tokens:")
                    print(f"    SUPPORTED ({allow[0]}): {probs[allow[0]].item():.6f}")
                    print(f"    REFUTED ({allow[1]}): {probs[allow[1]].item():.6f}")

                    # 找出其他高概率token
                    topk_values, topk_indices = torch.topk(logits, 5)
                    print("  Top 5 tokens by logits:")
                    for j, (val, idx_token) in enumerate(
                        zip(topk_values, topk_indices)
                    ):
                        token_str = tok.decode([idx_token.item()])
                        print(
                            f"    {j+1}. Token {idx_token.item()} ('{token_str}'): {val.item():.4f}"
                        )

            # 打印第一个批次的预测示例
            if i == 0 and debug_prefix:
                print(f"\n[DEBUG {debug_prefix}] First batch predictions:")
                for idx in range(min(3, len(batch))):
                    if isinstance(outs, dict) and "sequences" in outs:
                        sequences = outs.sequences
                        first_id = sequences[idx, inp_len].item()
                    else:
                        first_id = outs[idx, inp_len].item()

                    pred = parse_answer(first_id, allow[0])
                    print(f"  Sample {idx+1}: Token ID: {first_id}, Prediction: {pred}")
                    print(f"    Prompt: {batch[idx][:100]}...")

            # 收集预测结果
            if isinstance(outs, dict) and "sequences" in outs:
                sequences = outs.sequences
                for out in sequences:
                    first_id = out[inp_len].item()
                    pred = parse_answer(first_id, allow[0])
                    preds.append(pred)

                    # 检查是否所有预测都相同
                    if first_prediction is None:
                        first_prediction = pred
                    elif pred != first_prediction:
                        all_same_prediction = False
            else:
                for out in outs:
                    first_id = out[inp_len].item()
                    pred = parse_answer(first_id, allow[0])
                    preds.append(pred)

                    # 检查是否所有预测都相同
                    if first_prediction is None:
                        first_prediction = pred
                    elif pred != first_prediction:
                        all_same_prediction = False

        except Exception as e:
            print("Batch error:", e)
            print(traceback.format_exc())
            # 单条回退
            for j, p in enumerate(batch):
                try:
                    ids = tok(p, return_tensors="pt").to(device)
                    out = model.generate(
                        **ids,
                        max_new_tokens=1,
                        do_sample=False,
                        logits_processor=limiter,
                    )
                    first_id = out[0, ids["input_ids"].shape[1]].item()
                    pred = parse_answer(first_id, allow[0])
                    preds.append(pred)

                    # 检查是否所有预测都相同
                    if first_prediction is None:
                        first_prediction = pred
                    elif pred != first_prediction:
                        all_same_prediction = False
                except Exception as ie:
                    print("Single sample error:", ie)
                    preds.append("UNKNOWN")

    # 打印预测分布
    supported_count = preds.count("SUPPORTED")
    refuted_count = preds.count("REFUTED")
    unknown_count = preds.count("UNKNOWN")

    print(f"\n[DEBUG {debug_prefix}] Prediction distribution:")
    print(f"  SUPPORTED: {supported_count} ({supported_count/len(preds)*100:.1f}%)")
    print(f"  REFUTED: {refuted_count} ({refuted_count/len(preds)*100:.1f}%)")
    if unknown_count > 0:
        print(f"  UNKNOWN: {unknown_count} ({unknown_count/len(preds)*100:.1f}%)")

    # 警告如果所有预测都相同
    if all_same_prediction and len(preds) > 1:
        print(f"\n[WARNING] ALL predictions are the same: {first_prediction}")
        print("This suggests a potential issue with token identification or scoring.")

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
    # 如果使用了LoRA权重，在文件名中添加标识
    short = model_name.split("/")[-1] + ("_lora" if args.lora else "")
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
    # 如果使用了LoRA权重，在文件名中添加标识
    short = model_name.split("/")[-1] + ("_lora" if args.lora else "")
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
    global args  # 使args全局可访问，用于文件名
    args = parser.parse_args()

    df = load_data(args.data_path)
    orig_p, adv_p, skipped, valid = construct_prompts(df)

    tok, model = load_model(args.model, args.lora, args.token)  # 传递LoRA参数

    print("\nClassifying original …")
    o_preds = classify(tok, model, orig_p, args.batch_size, debug_prefix="ORIG")
    print("\nClassifying adversarial …")
    a_preds = classify(tok, model, adv_p, args.batch_size, debug_prefix="ADV")

    # 比较原始样本和对抗样本的预测
    flipped = sum(1 for o, a in zip(o_preds, a_preds) if o != a)
    print(
        f"\n[DEBUG] 直接比较: 总共 {flipped} 个样本的预测发生了翻转 ({flipped/len(o_preds)*100:.1f}%)"
    )

    # 显示一些翻转的例子
    if flipped > 0:
        print("[DEBUG] 翻转样本示例:")
        flipped_indices = [
            i for i, (o, a) in enumerate(zip(o_preds, a_preds)) if o != a
        ]
        for idx in flipped_indices[: min(3, len(flipped_indices))]:
            print(f"  样本 {idx+1}: {o_preds[idx]} -> {a_preds[idx]}")
            print(f"    原始: {orig_p[idx][:100]}...")
            print(f"    对抗: {adv_p[idx][:100]}...")
    else:
        print("[WARNING] 没有样本的预测发生翻转！这可能表明标记识别或评分存在问题。")

    valid_df = df.drop(index=skipped).reset_index(drop=True)
    res_df = compare_results_with_accuracy(
        valid_df, o_preds, a_preds, valid, args.model, args.output_dir
    )
    export_incorrect_predictions(res_df, args.model, args.output_dir)

    # 保存完整 CSV
    # 如果使用了LoRA权重，在文件名中添加标识
    model_name_with_suffix = args.model.split("/")[-1] + ("_lora" if args.lora else "")
    full_path = os.path.join(
        args.output_dir, model_name_with_suffix + "_full_results.csv"
    )
    res_df.to_csv(full_path, index=False)
    print(f"Full csv saved to {full_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("程序执行出错:", e)
        traceback.print_exc()
