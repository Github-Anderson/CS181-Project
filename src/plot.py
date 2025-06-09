import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 原始数据
data = {
    "G":   ["59/41", "73/27", "80/20", "25/25", "82/18", "54/46"],
    "M":   ["0/100", "0/100", "0/100", "15/35", "14/74", "0/100"],
    "MLS": ["0/100", "0/100", "0/100", "16/34", "23/63", "0/100"],
    "MCTS":["15/35","26/24","25/25","25/25","35/15","33/17"],
    "AQL": ["85/3", "60/40","58/42","5/45", "53/47","41/37"],
    "NAQL":["94/0","100/0","100/0","10/40","80/8","50/50"],
}

# 创建 DataFrame 存储字符串和计算数值
df_str = pd.DataFrame(data, index=["G", "M", "MLS", "MCTS", "AQL", "NAQL"])
left = df_str.applymap(lambda x: int(x.split('/')[0]))
right = df_str.applymap(lambda x: int(x.split('/')[1]))

# 计算差值 (左胜率 - 右胜率)
diff = (left - right) / (left + right)

# 转置矩阵，使热力图行列对应原表格
diff_t = diff.T

# 绘制热力图
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(diff_t, cmap='RdYlGn_r', vmin=-1, vmax=1)

# 设置坐标轴标签
ax.set_xticks(np.arange(diff_t.shape[1]))
ax.set_yticks(np.arange(diff_t.shape[0]))
ax.set_xticklabels(diff_t.columns)
ax.set_yticklabels(diff_t.index)

# 在每个单元格中添加文字注释
for i in range(diff_t.shape[0]):
    for j in range(diff_t.shape[1]):
        ax.text(j, i, df_str.iat[j, i],
                ha="center", va="center", 
                color="white" if abs(diff_t.iat[i, j]) > 0.5 else "black")

# 添加颜色条
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Left Win Rate - Right Win Rate')

# 美化并显示图像
plt.title('Hotmap of Win Rate Differences')
plt.tight_layout()
plt.show()
