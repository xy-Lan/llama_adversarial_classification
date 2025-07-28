import pandas as pd
import os

# --- Configuration ---
KEY_COLUMN = 'adversarial_samples'
COMPARISON_FILE = 'flip_data/comparison_output.csv'
MISTRAL_FILE = 'flip_data/mistral7b_lora_flipped_mp_correct.csv'
QWEN_FILE = 'flip_data/qwen_lora_flipped_mp_correct.csv'
OUTPUT_FILE = 'flip_data/comparison_output.csv' # Saving back to the same file

# --- Helper function to load CSV ---
def load_csv_if_exists(file_path, file_description):
    if not os.path.exists(file_path):
        print(f"警告: {file_description} 文件 '{file_path}' 未找到。将跳过此文件。")
        return None
    try:
        df = pd.read_csv(file_path)
        print(f"成功加载 {file_description} 文件: '{file_path}'，包含 {len(df)} 行。")
        if KEY_COLUMN not in df.columns:
            print(f"错误: 关键列 '{KEY_COLUMN}' 在 {file_description} 文件 '{file_path}' 中未找到。无法处理此文件。")
            return None
        return df
    except Exception as e:
        print(f"加载 {file_description} 文件 '{file_path}' 时出错: {e}")
        return None

def main():
    # 1. 加载现有的比较数据
    if not os.path.exists(COMPARISON_FILE):
        print(f"错误: 主要比较文件 '{COMPARISON_FILE}' 未找到。")
        print("此脚本期望该文件已存在，可能包含来自Llama模型的数据。")
        print("请确保文件存在，或者如果它应该被创建，请明确其初始状态。")
        return

    try:
        comparison_df = pd.read_csv(COMPARISON_FILE)
        print(f"成功从 '{COMPARISON_FILE}' 加载现有比较数据，包含 {len(comparison_df)} 行。")
    except Exception as e:
        print(f"加载现有比较文件 '{COMPARISON_FILE}' 时出错: {e}")
        return

    if KEY_COLUMN not in comparison_df.columns:
        print(f"错误: 关键列 '{KEY_COLUMN}' 在 '{COMPARISON_FILE}' 中未找到。")
        return

    # 确保 'frequency' 列存在且为字符串类型，处理潜在的NaN。
    # Llama脚本可能使用了 "0", "1", "01"。'm' 和 'q' 将被追加。
    if 'frequency' not in comparison_df.columns:
        comparison_df['frequency'] = '' # 如果缺失则初始化
    comparison_df['frequency'] = comparison_df['frequency'].astype(str).fillna('')

    # 使用 KEY_COLUMN 作为索引以提高查找和更新效率
    # 在重置索引时，也保留 KEY_COLUMN 作为一个常规列。
    # comparison_df_original_columns = comparison_df.columns.tolist() # 这行不再需要
    if KEY_COLUMN == comparison_df.index.name: 
         comparison_df = comparison_df.reset_index() 
    
    # 在设置为索引前，确保 KEY_COLUMN 没有重复值
    if comparison_df[KEY_COLUMN].duplicated().any():
        print(f"警告: 在 '{COMPARISON_FILE}' 的关键列 '{KEY_COLUMN}' 中发现重复值。")
        print("将保留第一次出现的值进行更新。请考虑清理文件。")
        comparison_df = comparison_df.drop_duplicates(subset=[KEY_COLUMN], keep='first')

    comparison_df = comparison_df.set_index(KEY_COLUMN, drop=False)

    # 用于存储尚不在 comparison_df 中的新行列表
    all_new_rows_data = []

    # 2. 处理 Mistral 数据，然后处理 Qwen 数据
    datasets_to_process = [
        (MISTRAL_FILE, 'm', "Mistral"),
        (QWEN_FILE, 'q', "Qwen")
    ]

    for file_path, model_char, model_name_desc in datasets_to_process:
        current_model_df = load_csv_if_exists(file_path, model_name_desc)
        if current_model_df is None:
            continue

        print(f"正在处理来自 '{file_path}' 的 {model_name_desc} 数据...")
        for _, row_series in current_model_df.iterrows(): # row_series is a pandas Series
            adv_sample_key = row_series[KEY_COLUMN]
            
            if pd.isna(adv_sample_key): # 更通用的NA检查
                print(f"因 '{KEY_COLUMN}' 中的键值缺失，跳过来自 {file_path} 的行。")
                continue

            if adv_sample_key in comparison_df.index:
                # 样本已存在: 更新其 'frequency'
                current_freq_val = comparison_df.loc[adv_sample_key, 'frequency']
                if model_char not in str(current_freq_val):
                    comparison_df.loc[adv_sample_key, 'frequency'] = str(current_freq_val) + model_char
                
                # 同时，使用当前模型行中的数据更新/添加其他列
                for col_name, col_value in row_series.items():
                    if col_name == KEY_COLUMN or col_name == 'frequency': # 跳过键和我们管理的frequency列
                        continue
                    
                    # .loc 赋值会自动创建新列（如果尚不存在），并为其他行填充NaN
                    comparison_df.loc[adv_sample_key, col_name] = col_value
            else:
                # 样本是全新的: 准备其数据以供添加
                new_row_dict = row_series.to_dict()
                new_row_dict['frequency'] = model_char
                all_new_rows_data.append(new_row_dict)
        print(f"已处理来自 '{file_path}' 的 {model_name_desc} 数据。")

    # 重置 comparison_df 的索引，使 KEY_COLUMN 再次成为常规列，以便进行 concat
    comparison_df = comparison_df.reset_index(drop=True) # drop=True 因为 KEY_COLUMN 已经作为普通列存在

    # 3. 将所有新行添加到 comparison_df
    if all_new_rows_data:
        new_rows_df = pd.DataFrame(all_new_rows_data)
        # Concatenate。这将对齐列，并在必要时添加NaN。
        comparison_df = pd.concat([comparison_df, new_rows_df], ignore_index=True, sort=False)
        print(f"已向比较数据中添加 {len(new_rows_df)} 条新的独立样本。")
    
    # 确保 concat 后所有行的 'frequency' 都是字符串类型
    comparison_df['frequency'] = comparison_df['frequency'].astype(str).fillna('')

    # 4. 为所有行重新计算 'frequency_count'
    # 此计数是 'frequency' 字符串中不同字符的数量。
    comparison_df['frequency_count'] = comparison_df['frequency'].apply(lambda f_str: len(set(f_str)))

    # 5. 按 'frequency_count' 降序排列 DataFrame
    comparison_df = comparison_df.sort_values(by='frequency_count', ascending=False)
    
    # 确保原始列加上 frequency 和 frequency_count 存在，来自新数据的其他列也保留。
    # 输出时没有特定的列选择，意味着保留所有合并的列。

    # 6. 保存更新后的 DataFrame
    try:
        comparison_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n成功更新数据并保存至 '{OUTPUT_FILE}'。")
        print(f"文件现在包含 {len(comparison_df)} 行。")
        print(f"文件已按 'frequency_count' (降序) 排序。")
        print(f"相关样本的 'frequency' 列包含了 Mistral 的 'm' 和 Qwen 的 'q'。")
    except Exception as e:
        print(f"保存更新数据至 '{OUTPUT_FILE}' 时出错: {e}")

if __name__ == "__main__":
    main() 