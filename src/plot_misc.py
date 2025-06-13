import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# 尝试设置更美观的字体 (如果系统支持)
# try:
#     plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans'] # 优先使用 Arial Unicode MS
#     plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
# except:
#     print("Arial Unicode MS not found, using default sans-serif font.")
#     pass


# 原始数据: key是后手 (Gote), index对应列表中的先手 (Sente)
# data = {
#     "G":   ["59/41", "0/100", "0/100", "15/35", "85/3", "94/0"],
#     "M":   ["73/27", "0/100", "0/100", "26/24", "60/40", "100/0"],
#     "MLS": ["80/20", "0/100", "0/100", "25/25", "58/42", "100/0"],
#     "MCTS":["25/25", "15/35", "16/34", "25/25", "5/45", "10/40"],
#     "AQL": ["82/18", "14/74", "23/63", "35/15", "53/47", "80/8"],
#     "NAQL":["54/46", "0/100", "0/100", "33/17", "41/37", "50/50"],
# }

# 新数据：
data00 = {
    "G":   ["39/12/49", "53/0/47", "54/0/46"],
    "M":   ["69/0/31",  "0/0/100", "0/0/100"],
    "MLS": ["79/1/20",  "0/0/100", "0/0/100"],
}

data01 = {
    "G":   ["45/5/50", "50/0/50", "46/0/54"],
    "M":   ["77/0/23", "0/0/100", "0/0/100"],
    "MLS": ["74/1/25", "0/0/100", "0/0/100"],
}

data10 = {
    "G":   ["18/51/31", "89/5/6", "87/6/7"],
    "M":   ["63/0/37",   "0/0/100", "0/0/100"],
    "MLS": ["71/0/29",   "0/0/100", "0/0/100"],
}

data11 = {
    "G":   ["13/69/18", "57/22/21",  "58/26/16"],
    "M":   ["64/6/37", "0/0/100", "0/0/100"],
    "MLS": ["58/11/31", "0/0/100", "0/0/100"],
}

data21 = {
    "G":   ["0/100/0",   "12/74/14",  "13/77/10"],
    "M":   ["17/83/0",  "0/0/100",  "0/0/100"],
    "MLS": ["17/83/0",  "0/0/100",  "0/0/100"],
}

data22 = { # This data is already numeric
    "G":   [1.5018, 2.8983, 2.3230],
    "M":   [3.5511, float('inf'), float('inf')],
    "MLS": [2.3924, float('inf'), float('inf')],
}


sente_agents_new = ["G", "M", "MLS"]
gote_agents_new = ["G", "M", "MLS"]

# --- Font size parameters for subplots ---
title_fontsize_s = 10
axis_label_fontsize_s = 9
tick_label_fontsize_s = 8
cell_text_fontsize_s = 8
cbar_label_fontsize_m = 10 # For main colorbar

def plot_heatmap_subplot(ax, data_dict, sente_agents, gote_agents, title_str, 
                         tick_label_fs, cell_text_fs, axis_label_fs, title_fs):
    df_str = pd.DataFrame(data_dict, index=sente_agents, columns=gote_agents)
    
    def parse_score(x_str):
        parts = x_str.split('/')
        return int(parts[0]), int(parts[1]), int(parts[2]) # Sente_wins, Draws, Gote_wins

    parsed_scores = df_str.applymap(parse_score)
    
    sente_wins_df = parsed_scores.applymap(lambda x: x[0])
    draws_df = parsed_scores.applymap(lambda x: x[1])
    gote_wins_df = parsed_scores.applymap(lambda x: x[2])

    total_games_df = sente_wins_df + draws_df + gote_wins_df
    
    diff_df_numerator = sente_wins_df - gote_wins_df
    
    # Calculate diff = (Sente_wins - Gote_wins) / total_games, handle division by zero
    diff_values = np.where(total_games_df.values == 0, 0.0, 
                           diff_df_numerator.values / total_games_df.values)
    diff_df = pd.DataFrame(diff_values, index=sente_agents, columns=gote_agents)

    diff_percentage = ((diff_df + 1) / 2 * 100).round(1)

    im = ax.imshow(diff_df, cmap='RdYlGn_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(np.arange(len(gote_agents)))
    ax.set_yticks(np.arange(len(sente_agents)))
    ax.set_xticklabels(gote_agents, fontsize=tick_label_fs)
    ax.set_yticklabels(sente_agents, fontsize=tick_label_fs)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Gote', fontsize=axis_label_fs, labelpad=5)
    ax.set_ylabel('Sente', fontsize=axis_label_fs, labelpad=5)

    ax.set_xticks(np.arange(len(gote_agents)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(sente_agents)+1)-.5, minor=True)
    ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False, top=False, right=False)

    for i in range(len(sente_agents)):
        for j in range(len(gote_agents)):
            cell_value = diff_df.iat[i, j]
            text_color = "white" if abs(cell_value) > 0.6 else "black"
            
            percentage_text = f"{diff_percentage.iat[i, j]:.1f}%"
            ax.text(j, i, percentage_text,
                    ha="center", va="center",
                    color=text_color, fontsize=cell_text_fs, weight='normal')
    
    ax.set_title(title_str, fontsize=title_fs, pad=4)
    return im

def plot_table_subplot(ax, data_dict, sente_agents, gote_agents, title_str,
                       tick_label_fs, cell_text_fs, axis_label_fs, title_fs):
    df_num = pd.DataFrame(data_dict, index=sente_agents, columns=gote_agents)
    
    ax.clear() 

    ax.set_xticks(np.arange(len(gote_agents)))
    ax.set_yticks(np.arange(len(sente_agents)))
    ax.set_xticklabels(gote_agents, fontsize=tick_label_fs)
    ax.set_yticklabels(sente_agents, fontsize=tick_label_fs)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Gote', fontsize=axis_label_fs, labelpad=5)
    ax.set_ylabel('Sente', fontsize=axis_label_fs, labelpad=5)
    
    ax.invert_yaxis() 

    # Define colormap and normalization for cell colors
    inf_substitute_val = 10.0  # Value to use for 'inf' in color scale
    cmap = plt.cm.get_cmap('RdYlGn_r')
    norm = mcolors.Normalize(vmin=1.0, vmax=inf_substitute_val)

    # Set cell boundaries with same style as heatmap subplots
    ax.set_xticks(np.arange(len(gote_agents)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(sente_agents)+1)-.5, minor=True)
    ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False, top=False, right=False)

    for i in range(len(sente_agents)):
        for j in range(len(gote_agents)):
            val = df_num.iloc[i, j]
            
            # Determine cell color
            val_for_color = inf_substitute_val if val == float('inf') else val
            # Clamp value to be within norm's vmin and vmax for robust color mapping
            val_for_color = max(norm.vmin, min(norm.vmax, val_for_color)) 
            cell_bgcolor = cmap(norm(val_for_color))

            # Add colored rectangle as cell background
            rect = mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                      facecolor=cell_bgcolor, 
                                      edgecolor='none', # Remove cell border
                                      zorder=1)
            ax.add_patch(rect)

            # Determine text color for readability
            # Simple luminance check: if background is dark, use white text, else black.
            # Luminance = 0.299*R + 0.587*G + 0.114*B
            luminance = 0.299*cell_bgcolor[0] + 0.587*cell_bgcolor[1] + 0.114*cell_bgcolor[2]
            text_color = "white" if luminance < 0.5 else "black"

            text_val = ""
            if val == float('inf'):
                text_val = r"$+ \infty$"
            elif isinstance(val, float):
                text_val = f"{val:.4f}"  # 改为四位小数
            else:
                text_val = str(val)
            
            ax.text(j, i, text_val, ha="center", va="center", 
                    color=text_color, fontsize=cell_text_fs, zorder=2)
    
    ax.set_title(title_str, fontsize=title_fs, pad=4)

# --- Main plotting ---
fig, axes = plt.subplots(3, 2, figsize=(6, 8)) # Adjusted figsize for 3x2 layout

datasets_for_heatmap = [
    (data00, r"$\mathtt{jump\_scalar}$ = 1.0"),
    (data01, r"$\mathtt{jump\_scalar}$ = 1.2"),
    (data10, r"$\mathtt{jump\_scalar}$ = 1.5"),
    (data11, r"$\mathtt{jump\_scalar}$ = 2.0"),
    (data21, r"$\mathtt{jump\_scalar}$ = 5.0")
]

im_ref = None 

for i, (data, title) in enumerate(datasets_for_heatmap):
    row, col = divmod(i, 2) # Changed from divmod(i, 3) to divmod(i, 2)
    ax = axes[row, col]
    current_im = plot_heatmap_subplot(ax, data, sente_agents_new, gote_agents_new, title,
                                      tick_label_fs=tick_label_fontsize_s,
                                      cell_text_fs=cell_text_fontsize_s,
                                      axis_label_fs=axis_label_fontsize_s,
                                      title_fs=title_fontsize_s)
    if i == 0: 
        im_ref = current_im

# Plot table for data22
ax_table = axes[2, 1] # Last subplot in a 3x2 grid (0-indexed)
plot_table_subplot(ax_table, data22, sente_agents_new, gote_agents_new, r"Converged $\mathtt{jump\_scalar}$",
                   tick_label_fs=tick_label_fontsize_s,
                   cell_text_fs=cell_text_fontsize_s,
                   axis_label_fs=axis_label_fontsize_s,
                   title_fs=title_fontsize_s)

# Add a common colorbar for the heatmaps
if im_ref:
    # Adjust colorbar position for 3x2 layout - align bottom with second row
    # 计算第二排子图的下边缘位置，并调整颜色条高度
    cbar_ax = fig.add_axes([0.85, 0.38, 0.03, 0.5]) # [left, bottom, width, height]
    cbar = fig.colorbar(im_ref, cax=cbar_ax)
    cbar.set_label('Win Rate (Sente)', fontsize=cbar_label_fontsize_m, labelpad=5)
    cbar.ax.tick_params(labelsize=tick_label_fontsize_s)

    # 设置颜色条刻度和标签以显示0%到100%
    # 这假设颜色条的数据范围 (例如通过 imshow 的 vmin/vmax 设置) 对应于 -1 到 1
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

fig.suptitle(r'Impact of $\mathtt{jump\_scalar}$ on Agent Performance', fontsize=16, weight='bold')

# Adjust layout to prevent overlap and make space for colorbar/suptitle
# May need different hspace/wspace for 3x2
# 减小 top 值以为主标题留出更多空间，同时调整 hspace 优化间距
fig.subplots_adjust(left=0.1, right=0.82, bottom=0.05, top=0.85, hspace=0.4, wspace=0.3)


plt.savefig("jump_scalar.png", dpi=600) # Changed filename
plt.show()

# Old code removed:
# sente_agents = ["G", "M", "MLS", "MCTS", "AQL", "NAQL"]
# gote_agents = ["G", "M", "MLS", "MCTS", "AQL", "NAQL"]
# ... (rest of the old plotting code)