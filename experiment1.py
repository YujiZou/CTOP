import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

data = pd.read_excel('.\experiment1.xlsx', sheet_name='Sheet1',header = 1)

x = data.iloc[:90, 16]  
y4 =data.iloc[:90,13]  #
y3 =data.iloc[:90,10]  #
y2= data.iloc[:90,7]  #
y1 = data.iloc[:90,4]
y0 = data.iloc[:90,1]

y4_1 =data.iloc[:90,14]  #
y3_1 =data.iloc[:90,11]  #
y2_1= data.iloc[:90,8]  #
y1_1 = data.iloc[:90,5]
y0_1 = data.iloc[:90,2]

plt.figure(1)

fig, ax = plt.subplots(figsize=(15, 6),dpi = 1000)

ax.plot(x, y0, color='#2B9CD7', linestyle='-', linewidth=1.5, marker='*', markersize=9, label='LDMA1')
ax.plot(x, y1, color='#DB726B', linestyle='-', linewidth=1.5, marker='^', markersize=7, label='LDMA2')
ax.plot(x, y2, color='#2AA371', linestyle='-', linewidth=1.5, marker='o', markersize=7, label='LDMA3')
ax.plot(x, y3, color='#7B6FD0', linestyle='-', linewidth=1.5, marker='+', markersize=7, label='LDMA_OX')
ax.plot(x, y4, color='#E69F00', linestyle='-', linewidth=1.5, marker='s', markersize=7, label='LDMA3_ERX')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)


ax.yaxis.set_major_formatter('{:.1f}'.format)


plt.xlabel('Instances', fontsize=20)
plt.ylabel('Deviation to LDMA in $f_{best}$(%)',  fontsize=15)
plt.xticks( fontsize=15)
plt.yticks( fontsize=15)
ax.set_xticks(np.arange(0, 91, 5))  
ax.set_xlim(0, 91)  #

plt.yticks(np.arange(0, 7.5, 0.5))  

ax.grid(which='major', color='#d9d9d9', linestyle='--', linewidth=0.1)
ax.grid(which='minor', color='#eeeeee', linestyle='--', linewidth=0.05)

ax.tick_params(direction='in')
for spine in ax.spines.values():
    spine.set_linewidth(1.5)  

plt.xlim(xmin=0)


plt.legend(prop={'size': 15})

plt.show()


plt.figure(2)

fig, ax = plt.subplots(figsize=(15, 6),dpi = 1000)


#
ax.plot(x, y0_1, color='#2B9CD7', linestyle='-', linewidth=1.5, marker='*', markersize=9, label='LDMA1')
ax.plot(x, y1_1, color='#DB726B', linestyle='-', linewidth=1.5, marker='^', markersize=7, label='LDMA2')
ax.plot(x, y2_1, color='#2AA371', linestyle='-', linewidth=1.5, marker='o', markersize=7, label='LDMA3')
ax.plot(x, y3_1, color='#7B6FD0', linestyle='-', linewidth=1.5, marker='+', markersize=7, label='LDMA_OX')
ax.plot(x, y4_1, color='#E69F00', linestyle='-', linewidth=1.5, marker='s', markersize=7, label='LDMA3_ERX')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)



plt.xlabel('Instances', fontsize=20)
plt.ylabel('Deviation to LDMA in $f_{avg}$(%)',  fontsize=15)
plt.xticks( fontsize=15)
plt.yticks( fontsize=15)
ax.set_xticks(np.arange(0, 91, 5))  
ax.set_xlim(0, 91) 

plt.yticks(np.arange(0, 9.5, 0.5))  
ax.grid(which='major', color='#d9d9d9', linestyle='--', linewidth=0.1)
ax.grid(which='minor', color='#eeeeee', linestyle='--', linewidth=0.05)

ax.tick_params(direction='in')
for spine in ax.spines.values():
    spine.set_linewidth(1.5)  

plt.xlim(xmin=0)


plt.legend(prop={'size': 15})

plt.show()




plt.figure(3)

# ============================================================
# 1. 文件路径及纵坐标设置
# ============================================================

#file_path = r'D:\papers\CTOP\analyze.xlsx'

#output_set2 = r'D:\papers\CTOP\runtime2_boxplot_set2.eps'
#output_set3 = r'D:\papers\CTOP\runtime2_boxplot_set3.eps'

# None表示根据全部数据自动确定纵坐标范围
# 如果需要固定范围，例如-25到25，可以修改为：
# Y_MIN = -25
# Y_MAX = 25
Y_MIN = None
Y_MAX = None

# 纵坐标刻度间隔
Y_TICK_INTERVAL = 5


# ============================================================
# 2. 读取Excel数据
# ============================================================

#data = pd.read_excel(
 #   file_path,
#    sheet_name='sbts',
 #   header=0
#)

# 读取前90行的第40～44列
# Python的列索引从0开始
y3 = pd.to_numeric(
    data.iloc[:90, 3],
    errors='coerce'
)

y4 = pd.to_numeric(
    data.iloc[:90, 6],
    errors='coerce'
)

y5 = pd.to_numeric(
    data.iloc[:90, 9],
    errors='coerce'
)

y_o_a = pd.to_numeric(
    data.iloc[:90, 12],
    errors='coerce'
)

y_e_a = pd.to_numeric(
    data.iloc[:90, 15],
    errors='coerce'
)


# ============================================================
# 3. 算法名称
# ============================================================

algorithm_names = [
    'LDMA1',
    'LDMA2',
   'LDMA3',
   'LDMA-OX',
    'LDMA-ERX'
]

all_results = [
    y3,
    y4,
    y5,
    y_o_a,
    y_e_a
]


# ============================================================
# 4. 划分Set II和Set III
# ============================================================

# Set II：前60个算例
set2_results = [
    values.iloc[:60].dropna().to_numpy()
    for values in all_results
]

# Set III：后30个算例
set3_results = [
    values.iloc[60:90].dropna().to_numpy()
    for values in all_results
]


# ============================================================
# 5. 设置绘图格式
# ============================================================

plt.rcParams.update({
    # STIXGeneral由Matplotlib提供，EPS兼容性较好
    'font.family': 'STIXGeneral',
    'mathtext.fontset': 'stix',

    'font.size': 17,
    'axes.labelsize': 19,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 16,

    'axes.linewidth': 1.5,

    'xtick.direction': 'in',
    'ytick.direction': 'in',

    'xtick.major.width': 1.4,
    'ytick.major.width': 1.4,

    'xtick.major.size': 5,
    'ytick.major.size': 5,

    # 将TrueType字体嵌入EPS
    'ps.fonttype': 42,

    # 防止纵坐标负号显示异常
    'axes.unicode_minus': False
})


# ============================================================
# 6. 颜色设置
# ============================================================

colors = [
    '#4C72B0',
    '#55A868',
    '#C44E52',
    '#8172B2',
    '#CCB974'
]


def blend_with_white(color, alpha):
    """
    将指定颜色与白色混合。

    EPS不支持真正的透明效果，所以使用混合后的浅色
    模拟半透明，同时保持图形为矢量对象。
    """
    rgb = np.array(to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])

    return alpha * rgb + (1.0 - alpha) * white


# 箱体浅色
box_colors = [
    blend_with_white(color, 0.65)
    for color in colors
]

# 散点浅色
point_colors = [
    blend_with_white(color, 0.50)
    for color in colors
]


# ============================================================
# 7. 计算统一的纵坐标范围
# ============================================================

combined_values = np.concatenate(
    set2_results + set3_results
)

minimum_value = np.nanmin(combined_values)
maximum_value = np.nanmax(combined_values)

# 自动将纵坐标上下限取为刻度间隔的整数倍
automatic_lower = (
    np.floor(minimum_value / Y_TICK_INTERVAL)
    * Y_TICK_INTERVAL
)

automatic_upper = (
    np.ceil(maximum_value / Y_TICK_INTERVAL)
    * Y_TICK_INTERVAL
)

# 确保包含LDMA的0基准线
automatic_lower = min(automatic_lower, 0)
automatic_upper = max(automatic_upper, 0)

# 防止所有数据完全相同时没有纵坐标范围
if automatic_lower == automatic_upper:
    automatic_lower -= Y_TICK_INTERVAL
    automatic_upper += Y_TICK_INTERVAL

# 使用自动范围或用户指定的范围
lower_limit = (
    automatic_lower
    if Y_MIN is None
    else Y_MIN
)

upper_limit = (
    automatic_upper
    if Y_MAX is None
    else Y_MAX
)

common_ylim = (
    lower_limit,
    upper_limit
)

common_yticks = np.arange(
    lower_limit,
    upper_limit + 0.001,
    Y_TICK_INTERVAL
)


# ============================================================
# 8. 固定散点的水平扰动
# ============================================================

# 固定随机种子，保证每次生成图片时散点位置相同
rng = np.random.default_rng(2026)


# ============================================================
# 9. 定义箱线图绘制函数
# ============================================================

def draw_runtime_boxplot(results):
    fig, ax = plt.subplots(figsize=(11, 6.8), dpi = 1000)

    positions = np.arange(1, len(results) + 1)

    boxplot = ax.boxplot(
        results,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={'color': 'black', 'linewidth': 2.0},
        whiskerprops={'color': 'black', 'linewidth': 1.5},
        capprops={'color': 'black', 'linewidth': 1.5},
        boxprops={'color': 'black', 'linewidth': 1.5}
    )

    for box, color in zip(boxplot['boxes'], box_colors):
        box.set_facecolor(color)

    for position, values, color in zip(positions, results, point_colors):
        jitter = rng.normal(loc=0, scale=0.047, size=len(values))
        ax.scatter(
            position + jitter,
            values,
            s=30,
            color=color,
            edgecolors='none',
            zorder=3
        )

    ax.axhline(y=0, color='#404040', linestyle='--', linewidth=1.8, zorder=1)

    legend_handle = Line2D([0], [0], color='#404040', linestyle='--', linewidth=1.8)

    legend = ax.legend(
        handles=[legend_handle],
        labels=['LDMA baseline'],
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
        handlelength=2.5,
        handletextpad=0.7,
        borderaxespad=0.2,
        fontsize=16
    )

    legend.set_zorder(10)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        algorithm_names,
        rotation=12,
        horizontalalignment='right',
        fontfamily='STIXGeneral'
    )

    ax.set_ylabel(
        'Runtime difference relative to LDMA (%)',
        fontfamily='STIXGeneral'
    )

    ax.set_ylim(common_ylim)
    ax.set_yticks(common_yticks)
    ax.set_xlim(0.5, len(results) + 0.5)

    ax.yaxis.grid(True, linestyle=':', linewidth=0.9, color='#B8B8B8')
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.show()
    plt.close(fig)


# ============================================================
# 10. 分别生成两个EPS文件
# ============================================================

draw_runtime_boxplot(
   results=set2_results
)

draw_runtime_boxplot(
   results=set3_results
)