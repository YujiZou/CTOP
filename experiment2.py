

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_excel('.\experiment2.xlsx', sheet_name='Sheet1',header = 1)

x = data.iloc[:90, 5]  

y2= data.iloc[:90,1]  #
y1 = data.iloc[:90,3]


y5 = data.iloc[:90,2]  
y4 = data.iloc[:90,4]
#y3 = data.iloc[:90,2]

plt.figure(1)

fig, ax = plt.subplots(figsize=(15, 6),dpi = 1000)

ax.plot(x, y2, color='#2B9CD7', linestyle='-', linewidth=1.5, marker='*', markersize=9, label='LDMA1')
ax.plot(x, y1, color='#DB726B', linestyle='-', linewidth=1.5, marker='^', markersize=7, label='LDMA2')
#ax.plot(x, y2, color='#2AA371', linestyle='-', linewidth=1.5, marker='o', markersize=7, label='LDMA3')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)


ax.yaxis.set_major_formatter('{:.1f}'.format)


plt.xlabel('Instances', fontsize=20)
plt.ylabel('Deviation to LDMA in $f_{best}$(%)',  fontsize=15)
plt.xticks( fontsize=15)
plt.yticks( fontsize=15)
ax.set_xticks(np.arange(0, 91, 5))  
ax.set_xlim(0, 91)  #

plt.yticks(np.arange(-0.1, 0.5, 0.1))  

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
ax.plot(x, y5, color='#2B9CD7', linestyle='-', linewidth=1.5, marker='*', markersize=9, label='LDMA4')
ax.plot(x, y4, color='#DB726B', linestyle='-', linewidth=1.5, marker='^', markersize=7, label='LDMA5')
#ax.plot(x, y5, color='#2AA371', linestyle='-', linewidth=1.5, marker='o', markersize=7, label='LDMA3')


ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)


plt.xlabel('Instances', fontsize=20)
plt.ylabel('Deviation to LDMA in $f_{avg}$(%)',  fontsize=15)
plt.xticks( fontsize=15)
plt.yticks( fontsize=15)
ax.set_xticks(np.arange(0, 91, 5))  
ax.set_xlim(0, 91) 

plt.yticks(np.arange(-0.1, 0.1, 0.7))  
ax.grid(which='major', color='#d9d9d9', linestyle='--', linewidth=0.1)
ax.grid(which='minor', color='#eeeeee', linestyle='--', linewidth=0.05)

ax.tick_params(direction='in')
for spine in ax.spines.values():
    spine.set_linewidth(1.5)  

plt.xlim(xmin=0)


plt.legend(prop={'size': 15})

plt.show()