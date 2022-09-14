import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Color = ['#7079DE', '#FC6F68', '#FFB64D']
data = pd.read_csv('Times.csv')
Number = np.asarray(data.iloc[:, 0])
Times = np.asarray(data.iloc[:, 1])

data = pd.read_csv('factor.csv')
year = np.asarray(data.iloc[:, 0])
Totalcar = np.asarray(data.iloc[:, 2])
Highway_c = np.asarray(data.iloc[:, 10])
City_c = np.asarray(data.iloc[:, 11])
High = np.asarray(data.iloc[:, 12])
City = np.asarray(data.iloc[:, 13])

# 计算拥堵次数
Center = 95.088  # 系数中心点
Low = 86    # 系数最低点
Up = 106     # 系数最高点
Per = 5  # 饱和流占总拥堵的比例

Times_high = Highway_c * Center * Per * 60 * 24 / Totalcar / 10000
Times_city = City_c * Center * Per * 60 * 4 / Totalcar / 10000

s = []
for i in range(0,len(Number)):
    s = np.append(s, Number[i] * np.ones(Times[i]))

"""
Total = np.vstack((Times_high, Times_city))
#直接变成DataFrame为一整列，需要转置变成一行
df_T = pd.DataFrame(Total)
df = pd.DataFrame(df_T.values.T)
#写入csv文件，mode = 'a'表示只在末尾追加写入，但是可能写入重复数据，注意最后导入的时候要进行查重清洗
df.to_csv('Times_R.csv', index = False, header = False, mode = 'a')
"""


# 系数图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.1, right=0.95, left=0.07, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(10000 / 900, 6000 / 900)
#设置刻度线在内
plt.tick_params(labelsize=15,direction='in')
plt.hist(s, 30, color=Color[0], edgecolor='black', linewidth=0.3, width=2)
#plt.plot((3,3), (0, gvf[2]), linewidth=1.0, color='k', linestyle='--')
#plt.plot((0,3), (gvf[2], gvf[2]), linewidth=1.0, color='k', linestyle='--')
# 设置坐标轴
plt.xlabel("Response number", fontsize=15)
plt.ylabel("Frequency", fontsize=15)
# 仅开启y轴方向的坐标轴
#plt.grid(axis='y')
#设置坐标轴范围
plt.xlim((40, 120))
plt.ylim((0, 50))
# 仅开启y轴方向的坐标轴
plt.grid(axis='y')
#设置坐标轴刻度
plt.xticks(np.arange(40,121,20))
plt.yticks(np.arange(0,51,10))
# 保存
plt.savefig(r'D:\Fig25.png', dpi=900)
plt.close()


# 道路里程
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.1, right=0.95, left=0.07, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(10000 / 900, 6000 / 900)
#设置刻度线在内
plt.tick_params(labelsize=15,direction='in')
plt.bar(year, City / 100, color = Color[0], label='Urban road')
plt.bar(year, High / 100, color = Color[1], bottom=City/100, label='Highway')
# 设置坐标轴
plt.ylabel("Total mileage (10$^{6}$ km)", fontsize=15)
plt.xlabel("Year", fontsize=15)
#设置坐标轴范围
plt.xlim((2012.5, 2050.5))
plt.ylim((0, 10))
# 图例
plt.legend(fontsize=15, loc='upper left')
#设置坐标轴刻度
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
plt.yticks(np.arange(2,10.1,2))
# 仅开启y轴方向的坐标轴
plt.grid(axis='y')
# 保存
plt.savefig(r'D:\Fig26.png', dpi=900)
plt.close()




# 反应数量
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.1, right=0.95, left=0.07, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(10000 / 900, 6000 / 900)
#设置刻度线在内
plt.tick_params(labelsize=15,direction='in')
plt.bar(year, Times_city, color = Color[0], label='Urban road')
plt.bar(year, Times_high, color = Color[1], bottom=Times_city, label='Highway')
# 设置坐标轴
plt.ylabel("Response number (per day)", fontsize=15)
plt.xlabel("Year", fontsize=15)
#设置坐标轴范围
plt.xlim((2012.5, 2050.5))
plt.ylim((0, 40))
# 图例
plt.legend(fontsize=15, loc='upper left')
#设置坐标轴刻度
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
plt.yticks(np.arange(8,40.1,8))
# 仅开启y轴方向的坐标轴
plt.grid(axis='y')
# 保存
plt.savefig(r'D:\Fig27.png', dpi=900)
plt.close()



# 车辆数量
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.1, right=0.95, left=0.07, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(10000 / 900, 6000 / 900)
#设置刻度线在内
plt.tick_params(labelsize=15,direction='in')
plt.bar(year, Totalcar / 10000, color = Color[0])
# 设置坐标轴
plt.ylabel("Vehicle number (10$^{8}$)", fontsize=15)
plt.xlabel("Year", fontsize=15)
#设置坐标轴范围
plt.xlim((2012.5, 2050.5))
plt.ylim((0, 7))
#设置坐标轴刻度
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
plt.yticks(np.arange(1.4,7.1,1.4))
# 仅开启y轴方向的坐标轴
plt.grid(axis='y')
# 保存
plt.savefig(r'D:\Fig28.png', dpi=900)
plt.close()



# 拥堵里程
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.1, right=0.95, left=0.09, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(10000 / 900, 6000 / 900)
# 设置刻度线在内
plt.tick_params(labelsize=15,direction='in')
plt.bar(year, City_c / 10000 * 4, color = Color[0], label='Urban road')
plt.bar(year, Highway_c / 10000 * 24, color = Color[1], bottom= City_c / 10000 * 4, label='Highway')
# 设置坐标轴
plt.ylabel("Congestion mileage (10$^{4}$ km per day)", fontsize=15)
plt.xlabel("Year", fontsize=15)
# 设置坐标轴范围
plt.xlim((2012.5, 2050.5))
plt.ylim((0, 50))
# 图例
plt.legend(fontsize=15, loc='upper left')
# 设置坐标轴刻度
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
plt.yticks(np.arange(0, 51, 10))
# 仅开启y轴方向的坐标轴
plt.grid(axis='y')
# 保存
plt.savefig(r'D:\Fig29.png', dpi=900)
plt.close()