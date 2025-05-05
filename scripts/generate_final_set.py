import pandas as pd

# 1. 读取数据集
original_without_first_test = "data/adversarial_dataset_without_sampled.csv"
remaining_80_percent = "data/remaining_80_percent.csv"

# 加载数据
df_without_first_test = pd.read_csv(original_without_first_test)
df_remaining_80 = pd.read_csv(remaining_80_percent)

print(f"原始数据集(已移除第一个测试集)大小: {len(df_without_first_test)}")
print(f"之前保留的80%数据大小: {len(df_remaining_80)}")

# 2. 从df_without_first_test中移除df_remaining_80的内容
def row_to_tuple(row):
    return tuple(row.values)

# 将要移除的数据转换为元组集合
to_remove_tuples = set(df_remaining_80.apply(row_to_tuple, axis=1))

# 筛选主数据集，保留不在要移除数据中的行
mask = ~df_without_first_test.apply(lambda row: row_to_tuple(row) in to_remove_tuples, axis=1)
filtered_df = df_without_first_test[mask]

print(f"移除后的数据集大小: {len(filtered_df)}")

# 3. 从筛选后的数据中抽取20%作为第二个测试集
second_test_size = int(len(filtered_df) * 0.2)
second_test_df = filtered_df.sample(n=second_test_size, random_state=42)

print(f"第二个测试集大小: {len(second_test_df)}")

# 4. 创建最终训练集：包括剩余的filtered_df和df_remaining_80
remaining_filtered_df = filtered_df.drop(second_test_df.index)
final_train_df = pd.concat([remaining_filtered_df, df_remaining_80], ignore_index=True)

# 5. 检查并删除可能的重复行
final_train_df_no_duplicates = final_train_df.drop_duplicates()
duplicates_count = len(final_train_df) - len(final_train_df_no_duplicates)

print(f"最终训练集大小(可能包含重复): {len(final_train_df)}")
print(f"最终训练集大小(去除重复): {len(final_train_df_no_duplicates)}")
print(f"删除的重复行数: {duplicates_count}")

# 6. 保存结果
second_test_file = "data/second_test_set.csv"
final_train_file = "data/final_complete_training_set.csv"

second_test_df.to_csv(second_test_file, index=False)
final_train_df_no_duplicates.to_csv(final_train_file, index=False)

print(f"第二个测试集已保存到: {second_test_file}")
print(f"最终完整训练集已保存到: {final_train_file}")

# 7. 计算最终的数据分布
original_total = len(df_without_first_test) + len(df_remaining_80) - duplicates_count
test_total = len(second_test_df)
train_total = len(final_train_df_no_duplicates)

print("\n最终数据分布:")
print(f"第二个测试集: {test_total}条样本 ({test_total/(test_total+train_total)*100:.1f}%)")
print(f"最终训练集: {train_total}条样本 ({train_total/(test_total+train_total)*100:.1f}%)")