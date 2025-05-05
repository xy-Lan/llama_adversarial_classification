#!/usr/bin/env python3
# split_adv_dataset.py
# ----------------------------------------------------------
# 将 data/adversarial_dataset.csv -> train.csv + test.csv
# ----------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split    # 安装: pip install scikit-learn

# 1. 读取原始 CSV
df = pd.read_csv("./data/adversarial_dataset.csv")

# 2. 按 80/20 随机划分；random_state 固定可复现
train_df, test_df = train_test_split(
    df,
    test_size=0.20,          # 20 % 测试
    shuffle=True,
    random_state=42          # 换成别的数字即可得到不同切分
)

# 3. 保存到文件
train_df.to_csv("./data/train.csv", index=False)
test_df.to_csv("./data/test.csv",  index=False)

print(f"Saved: {len(train_df)} rows -> ./data/train.csv")
print(f"Saved: {len(test_df)}  rows -> ./data/test.csv")