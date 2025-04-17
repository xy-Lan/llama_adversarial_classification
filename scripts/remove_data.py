import pandas as pd

# 1. 读取原始数据集和采样数据集
original_file = "data/adversarial_dataset.csv"
sampled_file = "data/sampled_20_percent.csv"

original_df = pd.read_csv(original_file)
sampled_df = pd.read_csv(sampled_file)

print(f"原始数据集大小: {len(original_df)}")
print(f"采样数据集大小: {len(sampled_df)}")

# 2. 方法1：使用索引直接移除（如果sampled_df保留了原始索引）
# 注意：这种方法仅在采样时保留了原始索引的情况下有效
if 'index' in sampled_df.columns:
    # 获取采样数据的原始索引
    sampled_indices = sampled_df['index'].tolist()
    # 从原始数据中移除这些索引
    remaining_df = original_df.drop(sampled_indices)

# 3. 方法2：基于数据内容移除（更可靠的方法）
else:
    # 创建一个函数，将每行转换为可哈希的元组形式
    def row_to_tuple(row):
        return tuple(row.values)


    # 将采样数据转换为元组集合
    sampled_tuples = set(sampled_df.apply(row_to_tuple, axis=1))

    # 筛选原始数据，保留不在采样数据中的行
    mask = ~original_df.apply(lambda row: row_to_tuple(row) in sampled_tuples, axis=1)
    remaining_df = original_df[mask]

# 4. 保存结果
output_file = "data/adversarial_dataset_without_sampled.csv"
remaining_df.to_csv(output_file, index=False)

print(f"移除采样数据后的数据集大小: {len(remaining_df)}")
print(f"结果已保存到: {output_file}")

# 5. 验证总数是否正确
expected_size = len(original_df) - len(sampled_df)
actual_size = len(remaining_df)
print(f"预期大小: {expected_size}, 实际大小: {actual_size}")
if expected_size == actual_size:
    print("✓ 数据移除成功！")
else:
    print("⚠ 警告：移除后的数据大小与预期不符，可能有重复或无法匹配的数据。")