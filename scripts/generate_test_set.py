import pandas as pd

# 1. 读取两个测试集
first_test_file = "data/sampled_20_percent.csv"
second_test_file = "data/second_test_set.csv"

# 加载数据
first_test_df = pd.read_csv(first_test_file)
second_test_df = pd.read_csv(second_test_file)

print(f"第一个测试集大小: {len(first_test_df)}")
print(f"第二个测试集大小: {len(second_test_df)}")

# 2. 合并两个测试集
combined_test_df = pd.concat([first_test_df, second_test_df], ignore_index=True)

# 3. 检查并删除可能的重复行
combined_test_df_no_duplicates = combined_test_df.drop_duplicates()
duplicates_count = len(combined_test_df) - len(combined_test_df_no_duplicates)

print(f"合并后的测试集大小: {len(combined_test_df)}")
print(f"删除的重复行数: {duplicates_count}")
print(f"最终测试集大小: {len(combined_test_df_no_duplicates)}")

# 4. 保存最终测试集
final_test_file = "data/final_test_set.csv"
combined_test_df_no_duplicates.to_csv(final_test_file, index=False)

print(f"最终测试集已保存到: {final_test_file}")

# 5. 验证没有重复数据
if duplicates_count == 0:
    print("✓ 两个测试集没有重叠数据")
else:
    print(f"⚠ 发现{duplicates_count}行重复数据已被移除")