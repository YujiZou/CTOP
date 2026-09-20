from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.ticker import MultipleLocator
from openpyxl import load_workbook


# ============================================================
# 1. 文件路径
# ============================================================

INPUT_FILE = Path("./experiment4.xlsx")


# ============================================================
# 2. Excel中的数据范围
# ============================================================

# 第2行到第91行，对应90个困难算例
FIRST_DATA_ROW = 3
LAST_DATA_ROW = 91


# 一共10组数据
# 从第3列开始，每隔3列读取一组：
# 3, 6, 9, 12, ..., 30
DATA_COLUMNS = list(range(4, 32, 3))


# 前3组用于第一个图
CUSTOMER_LABELS = [
    "Add",
    "Swap",
    "Remove",
]

CUSTOMER_COLUMNS = DATA_COLUMNS[:3]


# 后7组用于第二个图
ROUTE_LABELS = [
    "Relocate",
    "2-opt",
    "Swap",
    "Node-arc swap",
    "Arc-arc swap",
    "2-opt*",
    "Swap*",
]

ROUTE_COLUMNS = DATA_COLUMNS[3:]


# ============================================================
# 3. 颜色
# ============================================================

CUSTOMER_COLOURS = [
    "#5B8CCB",
    "#65B779",
    "#D97878",
]

ROUTE_COLOURS = [
    "#5B8CCB",
    "#65B779",
    "#D97878",
    "#8D7CC3",
    "#D5B85A",
    "#64B8B3",
    "#B678AF",
]


# ============================================================
# 4. 绘图样式
# ============================================================

def configure_style():
    plt.rcParams.update(
        {
            # Python窗口显示清晰度
            "figure.dpi": 180,

            # 保存图片的分辨率
            "savefig.dpi": 1200,

            # 字体
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",

            # 字号
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,

            "axes.linewidth": 1.1,

            # EPS字体兼容设置
            "ps.useafm": True,
            "ps.fonttype": 3,
            "text.usetex": False,

            # 防止负号无法显示
            "axes.unicode_minus": False,
        }
    )


# ============================================================
# 5. 读取Excel数据
# ============================================================

def read_runtime_differences(
    worksheet,
    labels,
    columns
):
    results = {}

    for label, column in zip(labels, columns):

        values = []

        for row in range(
            FIRST_DATA_ROW,
            LAST_DATA_ROW + 1
        ):

            value = worksheet.cell(
                row=row,
                column=column
            ).value

            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Row {row}, column {column} "
                    f"does not contain a valid number: {value!r}"
                )

            if not np.isfinite(value):
                raise ValueError(
                    f"Row {row}, column {column} "
                    f"contains a non-finite value: {value!r}"
                )

            values.append(float(value))

        results[label] = np.asarray(values)

    return results


# ============================================================
# 6. EPS不支持透明度，因此使用浅色模拟透明效果
# ============================================================

def blend_with_white(colour, opacity):

    rgb = np.asarray(
        to_rgb(colour)
    )

    white = np.ones(3)

    return tuple(
        opacity * rgb
        + (1.0 - opacity) * white
    )


# ============================================================
# 7. 自动设置纵坐标范围
# ============================================================

def calculate_y_limits(values):

    all_values = np.concatenate(values)

    data_min = min(
        float(all_values.min()),
        0.0
    )

    data_max = max(
        float(all_values.max()),
        0.0
    )

    data_range = max(
        data_max - data_min,
        10.0
    )

    padding = 0.08 * data_range

    lower = 5.0 * np.floor(
        (data_min - padding) / 5.0
    )

    upper = 5.0 * np.ceil(
        (data_max + padding) / 5.0
    )

    return lower, upper


# ============================================================
# 8. 绘制箱线图
# ============================================================

def draw_runtime_boxplot(
    results,
    colours
):

    labels = list(
        results.keys()
    )

    values = [
        results[label]
        for label in labels
    ]

    positions = np.arange(
        1,
        len(labels) + 1
    )

    # 两张图使用相同尺寸
    fig, ax = plt.subplots(
        figsize=(10.0, 6.0),
        dpi=180
    )

    boxplot = ax.boxplot(
        values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,

        medianprops={
            "color": "black",
            "linewidth": 1.8,
        },

        whiskerprops={
            "color": "#555555",
            "linewidth": 1.3,
        },

        capprops={
            "color": "#555555",
            "linewidth": 1.3,
        },

        boxprops={
            "edgecolor": "#3F3F3F",
            "linewidth": 1.4,
        },
    )

    # --------------------------------------------------------
    # 箱体颜色
    # --------------------------------------------------------

    for patch, colour in zip(
        boxplot["boxes"],
        colours,
    ):

        patch.set_facecolor(
            blend_with_white(
                colour,
                opacity=0.68,
            )
        )

        patch.set_alpha(1.0)


    # --------------------------------------------------------
    # 固定随机种子
    # --------------------------------------------------------

    rng = np.random.default_rng(
        2026
    )


    # --------------------------------------------------------
    # 绘制散点
    # --------------------------------------------------------

    for position, data, colour in zip(
        positions,
        values,
        colours,
    ):

        jitter = rng.uniform(
            -0.105,
            0.105,
            size=len(data),
        )

        ax.scatter(
            position + jitter,
            data,
            s=25,

            color=blend_with_white(
                colour,
                opacity=0.42,
            ),

            alpha=1.0,
            edgecolors="none",
            zorder=3,
        )


    # --------------------------------------------------------
    # LDMA基准线
    # --------------------------------------------------------

    ax.axhline(
        y=0.0,
        color="#3F3F3F",
        linestyle="--",
        linewidth=1.3,
        label="LDMA baseline",
        zorder=2,
    )


    # --------------------------------------------------------
    # 自动设置纵坐标范围
    # --------------------------------------------------------

    lower, upper = calculate_y_limits(
        values
    )

    ax.set_ylim(
        lower,
        upper,
    )


    # 第一张图范围较大时用50间隔
    # 第二张图范围较小时用5间隔
    if upper - lower > 150:
        tick_interval = 50
    else:
        tick_interval = 5

    ax.yaxis.set_major_locator(
        MultipleLocator(
            tick_interval
        )
    )


    # --------------------------------------------------------
    # 纵坐标名称
    # --------------------------------------------------------

    ax.set_ylabel(
        "Runtime difference relative to LDMA (%)"
    )


    # --------------------------------------------------------
    # 横坐标
    # --------------------------------------------------------

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels,
        rotation=12,
        ha="right",
    )


    # --------------------------------------------------------
    # 网格线
    # --------------------------------------------------------

    ax.grid(
        axis="y",
        color="#D9D9D9",
        linestyle="--",
        linewidth=0.7,
        alpha=1.0,
    )

    ax.set_axisbelow(
        True
    )


    # --------------------------------------------------------
    # 图例
    # --------------------------------------------------------

    ax.legend(
        loc="upper right",
        frameon=False,
    )


    # --------------------------------------------------------
    # 两张图使用相同边距
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.24,
        top=0.96,
    )


# ============================================================
# 9. 主程序
# ============================================================

def main():

    configure_style()

    # --------------------------------------------------------
    # 检查Excel文件是否存在
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Excel file not found: "
            f"{INPUT_FILE}"
        )


    # --------------------------------------------------------
    # 打开Excel
    # --------------------------------------------------------

    workbook = load_workbook(
        INPUT_FILE,
        data_only=True,
        read_only=True,
    )

    worksheet = workbook[
        "Sheet1"
    ]


    # --------------------------------------------------------
    # 第一个图
    # 读取第3、6、9列
    # --------------------------------------------------------

    customer_results = (
        read_runtime_differences(
            worksheet,
            CUSTOMER_LABELS,
            CUSTOMER_COLUMNS,
        )
    )


    # --------------------------------------------------------
    # 第二个图
    # 读取第12、15、18、21、24、27、30列
    # --------------------------------------------------------

    route_results = (
        read_runtime_differences(
            worksheet,
            ROUTE_LABELS,
            ROUTE_COLUMNS,
        )
    )


    # --------------------------------------------------------
    # 绘制两个图
    # --------------------------------------------------------

    draw_runtime_boxplot(
        customer_results,
        CUSTOMER_COLOURS,
    )

    draw_runtime_boxplot(
        route_results,
        ROUTE_COLOURS,
    )


    # --------------------------------------------------------
    # 同时显示两个图
    # --------------------------------------------------------

    plt.show()


if __name__ == "__main__":
    main()