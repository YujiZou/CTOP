

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = '.\experiment3.xlsx'  # Excel 文件路径
#sheet_name1 = 'hyper3'  # 需要读取的表格名称

df = pd.read_excel(file_path, header=2)

# 提取数据
value1, time1 = df.iloc[:, 0], df.iloc[:,1]
value2, time2 = df.iloc[:, 2], df.iloc[:, 3]

value3, time3 = df.iloc[:, 4], df.iloc[:,5]
value4, time4 = df.iloc[:, 6], df.iloc[:, 7]

value5, time5 = df.iloc[:, 8], df.iloc[:,9]
value6, time6 = df.iloc[:, 10], df.iloc[:,11]

value7, time7 = df.iloc[:, 12], df.iloc[:,13]
value8, time8 = df.iloc[:, 14], df.iloc[:, 15]

# 创建图像
plt.figure(1)
fig, ax = plt.subplots(figsize=(10, 6), dpi=600)

# 绘制曲线
ax.plot(time1, value1, label="LDMA", linewidth=2.5, marker='o', markersize=6)
ax.plot(time2, value2, label="LDMA6", linewidth=2.5, marker='s', markersize=5)

# 设置刻度朝内
ax.tick_params(axis="both", direction="in")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Time (seconds)', fontsize=15)
plt.ylabel('Objective value',  fontsize=15)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)  # 设置边框线宽

# **让 Matplotlib 自动设定刻度**
plt.draw()  # 让 Matplotlib 计算刻度

# **获取 Matplotlib 真实的坐标范围**
x_min, x_max = ax.get_xlim()  
y_min, y_max = ax.get_ylim()

# **减少网格线数量**
num_xgrid = 15  # X 方向网格线数量
num_ygrid = 15  # Y 方向网格线数量

x_grid_lines = np.linspace(x_min, x_max, num_xgrid)
y_grid_lines = np.linspace(y_min, y_max, num_ygrid)

# **使用更浅的颜色**
light_gray =  "#D0D0D0"  # 更浅的灰色

# **只添加网格，不改变刻度**
for x in x_grid_lines:
    ax.axvline(x, color=light_gray, linestyle="--", linewidth=0.2)  # 竖直网格线
for y in y_grid_lines:
    ax.axhline(y, color=light_gray, linestyle="--", linewidth=0.2)  # 水平网格线

# 添加图例
ax.legend(loc="center right", prop={'size': 15})

# 显示图像
#plt.savefig('D:\\papers\\CTOP\\v1.0\\2setb32.eps', format='eps', dpi=1000, transparent=False)
plt.show()

plt.figure(2)
fig, ax = plt.subplots(figsize=(10, 6), dpi=600)

# 绘制曲线
ax.plot(time3, value3, label="LDMA", linewidth=2.5, marker='o', markersize=6)
ax.plot(time4, value4, label="LDMA6", linewidth=2.5, marker='s', markersize=5)

# 设置刻度朝内
ax.tick_params(axis="both", direction="in")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Time (seconds)', fontsize=15)
plt.ylabel('Objective value',  fontsize=15)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)  # 设置边框线宽

# **让 Matplotlib 自动设定刻度**
plt.draw()  # 让 Matplotlib 计算刻度

# **获取 Matplotlib 真实的坐标范围**
x_min, x_max = ax.get_xlim()  
y_min, y_max = ax.get_ylim()

# **减少网格线数量**
num_xgrid = 15  # X 方向网格线数量
num_ygrid = 15  # Y 方向网格线数量

x_grid_lines = np.linspace(x_min, x_max, num_xgrid)
y_grid_lines = np.linspace(y_min, y_max, num_ygrid)

# **使用更浅的颜色**
light_gray =  "#D0D0D0"  # 更浅的灰色

# **只添加网格，不改变刻度**
for x in x_grid_lines:
    ax.axvline(x, color=light_gray, linestyle="--", linewidth=0.2)  # 竖直网格线
for y in y_grid_lines:
    ax.axhline(y, color=light_gray, linestyle="--", linewidth=0.2)  # 水平网格线

# 添加图例
ax.legend(loc="center right", prop={'size': 15})

# 显示图像
#plt.savefig('D:\\papers\\CTOP\\v1.0\\2setb54.eps', format='eps', dpi=1000, transparent=False)
plt.show()

plt.figure(3)
fig, ax = plt.subplots(figsize=(10, 6), dpi=600)

# 绘制曲线
ax.plot(time5, value5, label="LDMA", linewidth=2.5, marker='o', markersize=6)
ax.plot(time6, value6, label="LDMA6", linewidth=2.5, marker='s', markersize=5)

# 设置刻度朝内
ax.tick_params(axis="both", direction="in")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Time (seconds)', fontsize=15)
plt.ylabel('Objective value',  fontsize=15)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)  # 设置边框线宽

# **让 Matplotlib 自动设定刻度**
plt.draw()  # 让 Matplotlib 计算刻度

# **获取 Matplotlib 真实的坐标范围**
x_min, x_max = ax.get_xlim()  
y_min, y_max = ax.get_ylim()

# **减少网格线数量**
num_xgrid = 15  # X 方向网格线数量
num_ygrid = 15  # Y 方向网格线数量

x_grid_lines = np.linspace(x_min, x_max, num_xgrid)
y_grid_lines = np.linspace(y_min, y_max, num_ygrid)

# **使用更浅的颜色**
light_gray =  "#D0D0D0"  # 更浅的灰色

# **只添加网格，不改变刻度**
for x in x_grid_lines:
    ax.axvline(x, color=light_gray, linestyle="--", linewidth=0.2)  # 竖直网格线
for y in y_grid_lines:
    ax.axhline(y, color=light_gray, linestyle="--", linewidth=0.2)  # 水平网格线

# 添加图例
ax.legend(loc="center right", prop={'size': 15})

# 显示图像
#plt.savefig('D:\\papers\\CTOP\\v1.0\\2setb90.eps', format='eps', dpi=1000, transparent=False)
plt.show()

plt.figure(4)
fig, ax = plt.subplots(figsize=(10, 6), dpi=600)

# 绘制曲线
ax.plot(time7, value7, label="LDMA", linewidth=2.5, marker='o', markersize=6)
ax.plot(time8, value8, label="LDMA6", linewidth=2.5, marker='s', markersize=5)

# 设置刻度朝内
ax.tick_params(axis="both", direction="in")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Time (seconds)', fontsize=15)
plt.ylabel('Objective value',  fontsize=15)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)  # 设置边框线宽

# **让 Matplotlib 自动设定刻度**
plt.draw()  # 让 Matplotlib 计算刻度

# **获取 Matplotlib 真实的坐标范围**
x_min, x_max = ax.get_xlim()  
y_min, y_max = ax.get_ylim()

# **减少网格线数量**
num_xgrid = 15  # X 方向网格线数量
num_ygrid = 15  # Y 方向网格线数量

x_grid_lines = np.linspace(x_min, x_max, num_xgrid)
y_grid_lines = np.linspace(y_min, y_max, num_ygrid)

# **使用更浅的颜色**
light_gray =  "#D0D0D0"  # 更浅的灰色

# **只添加网格，不改变刻度**
for x in x_grid_lines:
    ax.axvline(x, color=light_gray, linestyle="--", linewidth=0.2)  # 竖直网格线
for y in y_grid_lines:
    ax.axhline(y, color=light_gray, linestyle="--", linewidth=0.2)  # 水平网格线

# 添加图例
ax.legend(loc="center right", prop={'size': 15})

# 显示图像
#plt.savefig('D:\\papers\\CTOP\\v1.0\\3setb30.eps', format='eps', dpi=1000, transparent=False)
plt.show()
