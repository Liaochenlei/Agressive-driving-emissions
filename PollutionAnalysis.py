import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
import scipy.stats

# 对输出的众多数据进行分析

data = pd.read_csv('OutcomeSDAI.csv')

ID = np.asarray(data.iloc[:, 0])
SDAI = np.asarray(data.iloc[:, 1])
HC = np.asarray(data.iloc[:, 2]); CO = np.asarray(data.iloc[:, 3])
NOx = np.asarray(data.iloc[:, 4]); NH3 = np.asarray(data.iloc[:, 5])
CO2 = np.asarray(data.iloc[:, 6]); PM = np.asarray(data.iloc[:, 7])
labels = np.asarray(data.iloc[:, 8])

plt.rcParams['font.sans-serif'] = ['Arial']  # 指定默认字体，有Arial和Times New Roman

Color = ['#7079DE', '#FC6F68', '#FFB64D']
Dis = ['Calm', 'Neutral', 'Aggressive']


"""
# 图1，将二维数据分成m输出 
m = 25
a = np.min(SDAI)
b = np.max(SDAI)
step = (b - a) / m
S_other = np.arange(a - step / 2, 0, -step)
S_new = np.arange(a + step / 2, b + step / 2, step)
Times = np.zeros(m)
for i in range(0, len(SDAI)):
    flag = int((SDAI[i] - a) // step)
    Times[flag] = Times[flag] + 1



fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.9, bottom=0.2, right=0.9, left=0.2, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(3000 / 900, 2000 / 900)

# 设置坐标轴
plt.xlabel("SDAI (s)", fontsize=10)
plt.ylabel("Frequency", fontsize=10)
plt.tick_params(direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
#axs.spines['bottom'].set_linewidth(0.5)
#axs.spines['left'].set_linewidth(0.5)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
flag = np.array([int((2.417598823 - a) // step), int((4.109660256 - a) // step)])
plt.bar(S_new[0:flag[0]], Times[0:flag[0]], width=0.2, color = Color[0], label=Dis[0])
plt.bar(S_new[flag[0]:flag[1]], Times[flag[0]:flag[1]], width=0.2, color = Color[1], label=Dis[1])
plt.bar(S_new[flag[1]:len(S_new)+1], Times[flag[1]:len(S_new)+1], width=0.2, color = Color[2], label=Dis[2])
#plt.legend(fontsize=5, loc='upper right', frameon=False)
#设置坐标轴范围
plt.xlim((0, 8))
plt.ylim((0, 8))
#设置坐标轴刻度
my_x_ticks = np.arange(0, 9, 1)
my_y_ticks = np.arange(0, 9, 2)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
# 保存
plt.savefig(r'D:\Fig3.png', dpi=900)
plt.close()



#采用Kmeans进行聚类
x = SDAI.reshape(-1,1)
km = KMeans(n_clusters=3)
km.fit(x)
centers = km.cluster_centers_ # 两组数据点的中心点
group = km.labels_   # 每个数据点所属分组
print(centers)
print(group)


# 用相关系数计算相关性,公式为（X-MeanX)*(Y-MeanY)/(StdX*StdY)
# R_nlenth = np.sum(((Oil - np.mean(Oil)) / np.std(Oil)) * ((niglenth - np.mean(niglenth)) / np.std(niglenth)))

#查看聚类的点
for i in range(0, 3):
    plt.scatter(SDAI[group == i], np.ones(len(group[group == i])))
plt.show()
"""



#输出污染图
#对CO2聚类分析
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.9, bottom=0.2, right=0.9, left=0.2, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(3500 / 900, 2500 / 900)
#设置刻度线在内
plt.tick_params(labelsize=13, direction='in')
for i in range(1, 4):
    plt.scatter(SDAI[labels == i], CO2[labels == i], color=Color[i - 1], label=Dis[i - 1])
plt.xlabel("SDAI (s)", fontsize=13)
plt.ylabel('CO$_{2}$ emission (g)', fontsize=13)
#设置坐标轴范围
plt.ylim((0, 32))
#设置坐标轴刻度
plt.yticks(np.array([8, 16, 24, 32]))
#显示图例
#plt.legend(fontsize=12, loc='lower right')
#设置x坐标和刻度
plt.xlim((0, 10))
plt.xticks(np.array([0, 2, 4, 6, 8, 10]))
#进行线性拟合并给出参数
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(SDAI, CO2)
# 斜率, 截距，相关系数，假设检验，估计梯度的标准差
plt.plot([min(SDAI), max(SDAI)], [slope * min(SDAI) + intercept, slope * max(SDAI) + intercept], color='k')
plt.text(2, 24, 'R$^{2}$=%.4f'%r_value,ha = 'center',va = 'bottom',fontsize=12)
# 保存
plt.savefig(r'D:\Fig4.png', dpi=900)
plt.close()




#对CO聚类分析
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.9, bottom=0.2, right=0.9, left=0.2, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(3500 / 900, 2500 / 900)
#设置刻度线在内
plt.tick_params(labelsize=13, direction='in')
for i in range(1, 4):
    plt.scatter(SDAI[labels == i], CO[labels == i] * 10, color=Color[i - 1], label=Dis[i - 1])
plt.xlabel("SDAI (s)", fontsize=13)
plt.ylabel('CO emission (10$^{-1}$ g)', fontsize=13)
#设置坐标轴范围
plt.ylim((0, 2))
#设置坐标轴刻度
plt.yticks(np.array([0.5, 1.0, 1.5, 2.0]))
#显示图例
#plt.legend(fontsize=12, loc='lower right')
#设置x坐标和刻度
plt.xlim((0, 10))
plt.xticks(np.array([0, 2, 4, 6, 8, 10]))
#进行线性拟合并给出参数
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(SDAI, CO * 10)
# 斜率, 截距，相关系数，假设检验，估计梯度的标准差
plt.plot([min(SDAI), max(SDAI)], [slope * min(SDAI) + intercept, slope * max(SDAI) + intercept], color='k')
plt.text(2, 1.5, 'R$^{2}$=%.4f'%r_value,ha = 'center',va = 'bottom',fontsize=12)
# 保存
plt.savefig(r'D:\Fig5.png', dpi=900)
plt.close()


#对HC聚类分析
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.9, bottom=0.2, right=0.9, left=0.2, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(3500 / 900, 2500 / 900)
#设置刻度线在内
plt.tick_params(labelsize=13, direction='in')
for i in range(1, 4):
    plt.scatter(SDAI[labels == i], HC[labels == i] * 1000, color=Color[i - 1], label=Dis[i - 1])
plt.xlabel("SDAI (s)", fontsize=13)
plt.ylabel('HC emission (10$^{-3}$ g)', fontsize=13)
#设置坐标轴范围
plt.ylim((0, 4))
#设置坐标轴刻度
plt.yticks(np.array([1, 2, 3, 4]))
#显示图例
#plt.legend(fontsize=12, loc='lower right')
#设置x坐标和刻度
plt.xlim((0, 10))
plt.xticks(np.array([0, 2, 4, 6, 8, 10]))
#进行线性拟合并给出参数
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(SDAI, HC * 1000)
# 斜率, 截距，相关系数，假设检验，估计梯度的标准差
plt.plot([min(SDAI), max(SDAI)], [slope * min(SDAI) + intercept, slope * max(SDAI) + intercept], color='k')
plt.text(2, 3, 'R$^{2}$=%.4f'%r_value,ha = 'center',va = 'bottom',fontsize=12)
# 保存
plt.savefig(r'D:\Fig6.png', dpi=900)
plt.close()


#对NOx聚类分析
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.9, bottom=0.2, right=0.9, left=0.2, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(3500 / 900, 2500 / 900)
#设置刻度线在内
plt.tick_params(labelsize=13, direction='in')
for i in range(1, 4):
    plt.scatter(SDAI[labels == i], NOx[labels == i] * 100, color=Color[i - 1], label=Dis[i - 1])
plt.xlabel("SDAI (s)", fontsize=13)
plt.ylabel('NO$_{x}$ emission (10$^{-2}$ g)', fontsize=13)
#设置坐标轴范围
plt.ylim((0, 1.6))
#设置坐标轴刻度
plt.yticks(np.array([0.4, 0.8, 1.2, 1.6]))
#显示图例
#plt.legend(fontsize=12, loc='lower right')
#设置x坐标和刻度
plt.xlim((0, 10))
plt.xticks(np.array([0, 2, 4, 6, 8, 10]))
#进行线性拟合并给出参数
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(SDAI, NOx * 100)
# 斜率, 截距，相关系数，假设检验，估计梯度的标准差
plt.plot([min(SDAI), max(SDAI)], [slope * min(SDAI) + intercept, slope * max(SDAI) + intercept], color='k')
plt.text(2, 1.2, 'R$^{2}$=%.4f'%r_value, ha='center', va='bottom', fontsize=12)
# 保存
plt.savefig(r'D:\Fig7.png', dpi=900)
plt.close()


# 对NH3聚类分析
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.9, bottom=0.2, right=0.9, left=0.2, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(3500 / 900, 2500 / 900)
# 设置刻度线在内
plt.tick_params(labelsize=13, direction='in')
for i in range(1, 4):
    plt.scatter(SDAI[labels == i], NH3[labels == i] * 10000, color=Color[i - 1], label=Dis[i - 1])
plt.xlabel("SDPI (s)", fontsize=13)
plt.ylabel('NH$_{3}$ emission (10$^{-4}$ g)', fontsize=13)
# 设置坐标轴范围
plt.ylim((0, 8))
# 设置坐标轴刻度
plt.yticks(np.array([2, 4, 6, 8]))
# 显示图例
# plt.legend(fontsize=8, loc='lower right')
# 设置x坐标和刻度
plt.xlim((0, 8))
plt.xticks(np.array([0, 2, 4, 6, 8]))
# 进行线性拟合并给出参数
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(SDAI, NH3 * 10000)
# 斜率, 截距，相关系数，假设检验，估计梯度的标准差
plt.plot([min(SDAI), max(SDAI)], [slope * min(SDAI) + intercept, slope * max(SDAI) + intercept], color='k')
plt.text(2, 6, 'R$^{2}$=%.4f'%r_value, ha='center', va='bottom', fontsize=13)
# 保存
plt.savefig(r'D:\Fig8.png', dpi=900)
plt.close()



#对PM聚类分析
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.9, bottom=0.2, right=0.9, left=0.2, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(3500 / 900, 2500 / 900)
#设置刻度线在内
plt.tick_params(labelsize=13, direction='in')
for i in range(1, 4):
    plt.scatter(SDAI[labels == i], PM[labels == i] * 10000, color=Color[i - 1], label=Dis[i - 1])
plt.xlabel("SDAI (s)", fontsize=13)
plt.ylabel('PM emission (10$^{-4}$ g)', fontsize=13)
#设置坐标轴范围
plt.ylim((0, 6))
#设置坐标轴刻度
plt.yticks(np.array([1.6, 3.2, 4.8, 6.4]))
#显示图例
#plt.legend(fontsize=12, loc='lower right')
#设置x坐标和刻度
plt.xlim((0, 10))
plt.xticks(np.array([0, 2, 4, 6, 8, 10]))
#进行线性拟合并给出参数
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(SDAI, PM * 10000)
# 斜率, 截距，相关系数，假设检验，估计梯度的标准差
plt.plot([min(SDAI), max(SDAI)], [slope * min(SDAI) + intercept, slope * max(SDAI) + intercept], color='k')
plt.text(2, 4.8, 'R$^{2}$=%.4f'%r_value, ha='center', va='bottom', fontsize=12)
# 保存
plt.savefig(r'D:\Fig9.png', dpi=900)
plt.close()


"""
#计算总量聚类分析
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.9, bottom=0.2, right=0.9, left=0.2, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(3500 / 900, 2500 / 900)
#设置刻度线在内
plt.tick_params(labelsize=13, direction='in')
for i in range(1, 4):
    plt.scatter(SDAI[labels == i], CO[labels == i] * 10 + NOx[labels == i] * 10 + HC[labels == i] * 10 + PM10[labels == i] * 10 * 2.23, color=Color[i - 1], label=Dis[i - 1])
plt.xlabel("SDAI (s)", fontsize=13)
plt.ylabel('Total pollutants (10$^{-1}$ g)', fontsize=13)

#设置坐标轴范围
plt.ylim((0, 2))
#设置坐标轴刻度
plt.yticks(np.array([0, 0.5, 1.0, 1.5, 2.0]))
#显示图例
#plt.legend(fontsize=12, loc='lower right')
#设置x坐标和刻度
plt.xlim((0, 10))
plt.xticks(np.array([0, 2, 4, 6, 8, 10]))
#进行线性拟合并给出参数
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(SDAI, CO * 10 + NOx * 10 + HC * 10 + PM10 * 10 * 2.23)
# 斜率, 截距，相关系数，假设检验，估计梯度的标准差
plt.plot([min(SDAI), max(SDAI)], [slope * min(SDAI) + intercept, slope * max(SDAI) + intercept], color='k')
plt.text(2, 1.5, 'R$^{2}$=%.4f'%r_value, ha='center', va='bottom', fontsize=12)
# 保存
plt.savefig(r'D:\Fig24.png', dpi=900)
plt.close()
"""

