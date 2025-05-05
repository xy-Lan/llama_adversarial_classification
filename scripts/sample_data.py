import pandas as pd
import numpy as np

# 1. 读取CSV文件
file_path = "data/Llama-3.2-1B-Instruct_preserved_flipped_samples.csv"

# 2. 读取数据
df = pd.read_csv(file_path)

# 3. 计算需要抽取的样本数量（20%）
sample_size = int(len(df) * 0.2)

# 4. 随机抽取样本
sampled_df = df.sample(n=sample_size, random_state=42)  # random_state确保结果可重现

# 5. 保存抽样结果到新文件
sampled_output_path = "data/sampled_20_percent.csv"
sampled_df.to_csv(sampled_output_path, index=False)

# 6. 保存剩余的80%样本
remaining_df = df.drop(sampled_df.index)
remaining_output_path = "data/remaining_80_percent.csv"
remaining_df.to_csv(remaining_output_path, index=False)

print(f"原始数据集大小: {len(df)}")
print(f"抽样数据集大小: {len(sampled_df)} ({len(sampled_df)/len(df)*100:.1f}%)")
print(f"剩余数据集大小: {len(remaining_df)} ({len(remaining_df)/len(df)*100:.1f}%)")
print(f"抽样结果已保存到: {sampled_output_path}")
print(f"剩余数据已保存到: {remaining_output_path}")