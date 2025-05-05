import pandas as pd

# ❶ 读文件
df = pd.read_csv("./data/adversarial_dataset.csv")

# ❷ 提取键
def get_key(text):
    return text.split(".", 1)[0].strip().lower()

df["merge_key"] = df["original_samples"].apply(get_key)

# ❸ 记录原始行号（从 1 开始）
df["orig_row"] = df.index + 1

# ❹ 定义每列怎样聚合
agg_dict = {
    "original_samples": lambda s: "|||".join(s.unique()),
    "adversarial_samples": lambda s: "|||".join(s.unique()),
    "orig_row": lambda s: list(s)
}

# ❺ 分组聚合
merged = df.groupby("merge_key", as_index=False).agg(agg_dict)
merged = merged.rename(columns={"orig_row": "line_numbers"})

# ❻ 保存
merged.to_csv("./data/merged_dataset.csv", index=False)
print("Done! 已保存为 merged_dataset.csv")
