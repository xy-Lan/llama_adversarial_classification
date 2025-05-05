import pandas as pd

# 读取 CSV 文件
adv_df = pd.read_csv('./data/adversarial_dataset.csv')
merged_df = pd.read_csv('./data/merged_dataset.csv')

# 创建查找字典：original_samples → correctness
merged_lookup = dict(zip(merged_df['original_samples'], merged_df['correctness']))

# 填补 correctness 为空的行
adv_df['correctness'] = adv_df.apply(
    lambda row: merged_lookup.get(row['original_samples'], row['correctness']),
    axis=1
)

# 保存结果
adv_df.to_csv('./data/adversarial_dataset_corrected.csv', index=False)
