import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 尝试设置更美观的字体 (如果系统支持)
# try:
#     plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans'] # 优先使用 Arial Unicode MS
#     plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
# except:
#     print("Arial Unicode MS not found, using default sans-serif font.")
#     pass


# 原始数据: key是后手 (Gote), index对应列表中的先手 (Sente)
data = {
    "G":   ["59/41", "73/27", "80/20", "25/25", "82/18", "54/46"],
    "M":   ["0/100", "0/100", "0/100", "15/35", "14/74", "0/100"],
    "MLS": ["0/100", "0/100", "0/100", "16/34", "23/63", "0/100"],
    "MCTS":["15/35","26/24","25/25","25/25","35/15","33/17"],
    "AQL": ["85/3", "60/40","58/42","5/45", "53/47","41/37"],
    "NAQL":["94/0","100/0","100/0","10/40","80/8","50/50"],
}

sente_agents = ["G", "M", "MLS", "MCTS", "AQL", "NAQL"]
gote_agents = ["G", "M", "MLS", "MCTS", "AQL", "NAQL"]

df_str = pd.DataFrame(data, index=sente_agents)
df_str = df_str[gote_agents]

left_wins = df_str.applymap(lambda x: int(x.split('/')[0]))
right_wins = df_str.applymap(lambda x: int(x.split('/')[1]))

total_games = left_wins + right_wins
diff = np.where(total_games == 0, 0, (left_wins - right_wins) / total_games)
diff_df = pd.DataFrame(diff, index=sente_agents, columns=gote_agents)

# --- 绘图参数调整 ---
title_fontsize = 18 # 增大标题字号
axis_label_fontsize = 12
tick_label_fontsize = 10
cell_text_fontsize = 10 # 增大单元格内文字字号
cbar_label_fontsize = 11
figure_size = (11, 9)

fig, ax = plt.subplots(figsize=figure_size)
im = ax.imshow(diff_df, cmap='RdYlGn_r', vmin=-1, vmax=1, aspect='auto')

# 设置坐标轴标签和刻度
ax.set_xticks(np.arange(len(gote_agents)))
ax.set_yticks(np.arange(len(sente_agents)))
ax.set_xticklabels(gote_agents, fontsize=tick_label_fontsize)
ax.set_yticklabels(sente_agents, fontsize=tick_label_fontsize)

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xlabel('Second Player (Gote)', fontsize=axis_label_fontsize, labelpad=10)
ax.set_ylabel('First Player (Sente)', fontsize=axis_label_fontsize, labelpad=10)

# 添加细微的网格线
ax.set_xticks(np.arange(len(gote_agents)+1)-.5, minor=True)
ax.set_yticks(np.arange(len(sente_agents)+1)-.5, minor=True)
ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5, alpha=0.3)
ax.tick_params(which="minor", bottom=False, left=False)


# 在每个单元格中添加文字注释
for i in range(len(sente_agents)):
    for j in range(len(gote_agents)):
        cell_value = diff_df.iat[i, j]
        text_color = "white" if abs(cell_value) > 0.6 else "black"
        
        if -0.1 < cell_value < 0.1 and abs(cell_value) <=0.6 :
             text_color = "black"

        ax.text(j, i, f"$\it{{{df_str.iat[i, j]}}}$",
                ha="center", va="center",
                color=text_color, fontsize=cell_text_fontsize, weight='normal')

# 添加颜色条
cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.05)
cbar.set_label('Win Rate Difference (Sente - Gote)', fontsize=cbar_label_fontsize, labelpad=10)
cbar.ax.tick_params(labelsize=tick_label_fontsize-1)

# 设置标题
plt.title('Heatmap of Win Rate Differences (Sente vs Gote)', fontsize=title_fontsize, pad=25, weight='bold') # 增加 weight='bold'

# 调整布局并保存
plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
plt.savefig("win_rate_heatmap_further_enhanced.png", dpi=300, bbox_inches='tight') # 修改保存文件名
plt.show()