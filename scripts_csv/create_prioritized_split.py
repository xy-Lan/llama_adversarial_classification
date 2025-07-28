import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split

def create_prioritized_split(args):
    # 1. 加载主要数据文件
    try:
        main_df = pd.read_csv(args.input_csv)
        print(f"成功加载主要数据文件: '{args.input_csv}'，包含 {len(main_df)} 行。")
    except FileNotFoundError:
        print(f"错误: 主要输入文件 '{args.input_csv}' 未找到。")
        return
    except Exception as e:
        print(f"加载主要输入文件 '{args.input_csv}' 时出错: {e}")
        return

    if args.key_column not in main_df.columns:
        print(f"错误: 关键列 '{args.key_column}' 在主要输入文件 '{args.input_csv}' 中未找到。")
        return

    # 2. 加载优先样本数据文件 (comparison_output.csv)
    comparison_df = None
    if os.path.exists(args.comparison_csv):
        try:
            comparison_df = pd.read_csv(args.comparison_csv)
            if args.key_column not in comparison_df.columns:
                print(f"警告: 关键列 '{args.key_column}' 在优先样本文件 '{args.comparison_csv}' 中未找到。将忽略此文件。")
                comparison_df = None
            else:
                print(f"成功加载优先样本文件: '{args.comparison_csv}'，包含 {len(comparison_df)} 行。")
                # 假设 comparison_output.csv 已按 frequency_count 降序排列
        except Exception as e:
            print(f"加载优先样本文件 '{args.comparison_csv}' 时出错: {e}。将忽略此文件。")
            comparison_df = None
    else:
        print(f"警告: 优先样本文件 '{args.comparison_csv}' 未找到。将执行标准随机划分。")

    # 3. 确定必须放入训练集的样本键
    must_be_train_keys = set()
    if comparison_df is not None and not comparison_df.empty:
        if args.top_n_comparison > 0:
            # N.B. comparison_df is assumed to be sorted by frequency_count descending
            must_be_train_keys = set(comparison_df[args.key_column].head(args.top_n_comparison).tolist())
            print(f"将优先样本文件中的前 {args.top_n_comparison} 个样本 (共 {len(must_be_train_keys)} 个独立键) 强制放入训练集。")
        else: # top_n_comparison == 0 (默认) 表示所有 comparison_output.csv 中的样本
            must_be_train_keys = set(comparison_df[args.key_column].tolist())
            print(f"将优先样本文件中的所有样本 (共 {len(must_be_train_keys)} 个独立键) 强制放入训练集。")
    
    # 4. 从主要数据中分离出强制训练样本和待分配池
    # 确保即时在主数据中，这些强制样本也是唯一的（如果主数据本身有基于key_column的重复）
    # 如果主数据中，一个key对应多行，这些行都会被强制放入训练集
    must_be_train_df = main_df[main_df[args.key_column].isin(list(must_be_train_keys))].copy()
    
    # 实际被强制放入训练集的独立键 (可能小于 must_be_train_keys，如果某些键不在 main_df 中)
    actual_forced_keys_in_main_df = set(must_be_train_df[args.key_column].tolist())
    if len(actual_forced_keys_in_main_df) < len(must_be_train_keys):
        print(f"警告: 在 '{args.input_csv}' 中实际找到并强制分配到训练集的优先样本键有 {len(actual_forced_keys_in_main_df)} 个，少于请求的 {len(must_be_train_keys)} 个。")
    
    pool_for_splitting_df = main_df[~main_df[args.key_column].isin(list(actual_forced_keys_in_main_df))].copy()

    print(f"强制放入训练集的样本数: {len(must_be_train_df)}")
    print(f"进入随机分配池的样本数: {len(pool_for_splitting_df)}")

    # 5. 计算从待分配池中选取的训练集和测试集大小
    n_total_main = len(main_df)
    target_train_overall_size = round(n_total_main * (1 - args.test_size))
    
    n_already_in_train = len(must_be_train_df)
    num_needed_from_pool_for_train = max(0, target_train_overall_size - n_already_in_train)

    n_pool = len(pool_for_splitting_df)

    # 确保从池中选取的训练样本数不超过池的大小
    num_needed_from_pool_for_train = min(num_needed_from_pool_for_train, n_pool)
    
    num_needed_from_pool_for_test = n_pool - num_needed_from_pool_for_train

    print(f"目标总训练集大小: {target_train_overall_size}")
    print(f"已通过强制分配获得训练样本: {n_already_in_train}")
    print(f"需从分配池中获取的训练样本: {num_needed_from_pool_for_train}")
    print(f"需从分配池中获取的测试样本: {num_needed_from_pool_for_test}")

    # 6. 划分待分配池
    additional_train_df = pd.DataFrame(columns=main_df.columns)
    test_df_from_pool = pd.DataFrame(columns=main_df.columns)

    if n_pool > 0:
        if num_needed_from_pool_for_train == n_pool: # 池中所有样本都用于训练
            additional_train_df = pool_for_splitting_df.copy()
            # test_df_from_pool 保持为空
        elif num_needed_from_pool_for_test == n_pool: # 池中所有样本都用于测试
            test_df_from_pool = pool_for_splitting_df.copy()
            # additional_train_df 保持为空
        elif num_needed_from_pool_for_train > 0 or num_needed_from_pool_for_test > 0: # 需要分割
            # sklearn 的 test_size 参数是 *测试集* 的比例
            split_test_size_for_pool = num_needed_from_pool_for_test / n_pool
            additional_train_df, test_df_from_pool = train_test_split(
                pool_for_splitting_df, 
                test_size=split_test_size_for_pool, 
                random_state=args.random_state,
                # stratify=pool_for_splitting_df[args.stratify_column] if args.stratify_column else None # 可选的分层抽样
            )
    
    # 7. 合并形成最终的训练集和测试集
    final_train_df = pd.concat([must_be_train_df, additional_train_df], ignore_index=True)
    final_test_df = test_df_from_pool # test_df_from_pool 就是最终的测试集

    # 打乱顺序 (可选，但推荐)
    final_train_df = final_train_df.sample(frac=1, random_state=args.random_state).reset_index(drop=True)
    final_test_df = final_test_df.sample(frac=1, random_state=args.random_state).reset_index(drop=True)

    print(f"\n最终训练集大小: {len(final_train_df)} (目标: {target_train_overall_size})")
    print(f"最终测试集大小: {len(final_test_df)} (目标: {n_total_main - target_train_overall_size})")
    
    # 检查是否有样本同时存在于训练集和测试集 (基于key_column)
    if not final_train_df.empty and not final_test_df.empty:
        train_keys = set(final_train_df[args.key_column].unique())
        test_keys = set(final_test_df[args.key_column].unique())
        overlap = train_keys.intersection(test_keys)
        if overlap:
            print(f"严重警告: 训练集和测试集之间存在 {len(overlap)} 个重叠的键: {list(overlap)[:5]}...")
        else:
            print("检查通过: 训练集和测试集之间没有基于关键列的重叠。")

    # 8. 保存文件
    os.makedirs(args.output_dir, exist_ok=True)
    train_output_path = os.path.join(args.output_dir, args.train_filename)
    test_output_path = os.path.join(args.output_dir, args.test_filename)

    try:
        final_train_df.to_csv(train_output_path, index=False)
        print(f"训练集已保存至: {train_output_path}")
        final_test_df.to_csv(test_output_path, index=False)
        print(f"测试集已保存至: {test_output_path}")
    except Exception as e:
        print(f"保存输出文件时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从主要数据文件创建训练集和测试集，并优先将指定样本放入训练集。")
    parser.add_argument("--input_csv", type=str, default="data/adversarial_dataset_corrected.csv", help="主要输入数据文件的路径 (.csv)")
    parser.add_argument("--comparison_csv", type=str, default="flip_data/comparison_output.csv", help="包含优先样本列表的CSV文件路径 (应按频率降序排列)。")
    parser.add_argument("--output_dir", type=str, default="scripts_csv", help="保存输出 train.csv 和 test.csv 的目录。")
    parser.add_argument("--train_filename", type=str, default="train.csv", help="训练集输出文件名。")
    parser.add_argument("--test_filename", type=str, default="test.csv", help="测试集输出文件名。")
    parser.add_argument("--key_column", type=str, default="adversarial_samples", help="用于在不同文件中识别和匹配样本的列名。")
    parser.add_argument("--top_n_comparison", type=int, default=0, help="从 comparison_csv 文件顶部强制放入训练集的样本数量。0 表示所有 comparison_csv 中的样本。(默认: 0)")
    parser.add_argument("--test_size", type=float, default=0.2, help="测试集应占总数据集的比例。(默认: 0.2)")
    parser.add_argument("--random_state", type=int, default=42, help="用于随机操作的种子，以确保可复现性。(默认: 42)")
    # parser.add_argument("--stratify_column", type=str, default=None, help="(可选) 用于分层抽样的列名。") # 可选参数

    args = parser.parse_args()
    create_prioritized_split(args) 