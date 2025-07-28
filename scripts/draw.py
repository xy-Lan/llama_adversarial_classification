import matplotlib.pyplot as plt
import numpy as np

# 模型名称
models = ['LLaMA-1B', 'LLaMA-3B', 'LLaMA-8B', 'Mistral-7B', 'Qwen-7B']

# 各攻击方式的成功比例（百分数）
homoglyph = [76.67, 73.68, 74.07, 74.47, 75.00]
zero_width = [0.00, 10.53, 11.11, 10.64, 0.00]
word_split = [23.33, 15.79, 14.81, 14.89, 25.00]
random_change = [0, 0, 0, 0, 0]  # 都是0，但可保留用于图例完整性

# x 轴位置
x = np.arange(len(models))
width = 0.6

# 画图
fig, ax = plt.subplots(figsize=(10, 6))

# 堆叠条形图
p1 = ax.bar(x, homoglyph, width, label='Homoglyph Substitution')
p2 = ax.bar(x, zero_width, width, bottom=homoglyph, label='Zero-width / Punctuation')
bottom2 = np.array(homoglyph) + np.array(zero_width)
p3 = ax.bar(x, word_split, width, bottom=bottom2, label='Word Splitting')
bottom3 = bottom2 + np.array(word_split)
p4 = ax.bar(x, random_change, width, bottom=bottom3, label='Random / Boundary Change')

# 图示设置
ax.set_ylabel('Success Rate (%)')
ax.set_title('Breakdown of Character-Level Attack Success by Strategy')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 110)
ax.legend(loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


