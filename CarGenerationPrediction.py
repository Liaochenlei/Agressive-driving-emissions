import pandas as pd
from math import e
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import interpolate
import matplotlib.pyplot as plt


################################################
# 以下为补全和预测增量部分
# 以下为实际数值,GDP单位为美元，人口为万人
Year_old = np.arange(2012, 2021)
GDP = np.array([8532229986993, 9570406235659, 10475682920597, 11061553079871, 11233276536744, 12310409370894, 13894817549380, 14279937467431, 14722730697890])
Population = np.array([135404, 136072, 136782, 137462, 138271, 139008, 139538, 140005, 141178])
# 车辆增量，2013-2020年
Car = np.array([2017, 2188, 2385, 2752, 2813, 2673, 2578, 2424])

# 以下为补全预测功能
# UN的联合国预测(单位：k）,由于车辆数是没有台湾、香港、澳门的，所以要去掉
Year_use1 = np.arange(2020, 2061, 5)
# 2012-2020
Population_use = np.array([1407360.669, 1425493.663, 1431578.343, 1428292.369, 1416507.141, 1397321.995, 1371112.769, 1339066.038, 1303275.936])

# PWC经济预测（单位：万亿），经济分为in PPP and in MER
Year_use2 = np.array([2016, 2020, 2025, 2030, 2035, 2040, 2045, 2050])
GDP_use_PPP = np.array([21.3, 26.9, 32.7, 38.0, 42.7, 47.4, 52.9, 58.5])
GDP_use_MER = np.array([11.4, 16.0, 21.3, 26.5, 31.4, 36.6, 43.0, 49.9])
# 三次样条补全，多一年用于预测增长率
Year1 = np.arange(2020, 2051)
Year2 = np.arange(2015, 2051)
tck_Population = interpolate.splrep(Year_use1, Population_use, k=3)
tck_GDP = interpolate.splrep(Year_use2, GDP_use_PPP, k=3)
Population_new = interpolate.splev(Year1, tck_Population)
GDP_out = interpolate.splev(Year2, tck_GDP)

"""
# 画图验证
plt.plot(Year_use1, Population_use, 'o', Year1, Population_new)
plt.show()
plt.plot(Year_use2, GDP_use_PPP, 'o', Year2, GDP_out)
plt.show()
"""

# 合并2015年到2020年与预测的人口数，单位统一为k
Population_out = np.hstack((Population[3:8] * 10, Population_new[:]))

# 计算GDP和人口增长率,2021-2050
GDP_speed = np.zeros(35)
Population_speed = np.zeros(35)
for i in range(0, 35):
    GDP_speed[i] = (GDP_out[i + 1] - GDP_out[i]) / GDP_out[i]
    Population_speed[i] = (Population_out[i + 1] - Population_out[i]) / Population_out[i]


# 输入车辆新增量，2016-2020, 单位为万,
Car_increase_real = np.array([2752, 2813, 2673, 2578, 2424])

# 拟合增量
X = np.array(np.vstack([GDP_speed[0:5], Population_speed[0:5]])).reshape(5, 2)
Y = Car_increase_real.reshape(5, 1)

# 进行二元线性回归
model = LinearRegression()
model.fit(X, Y)
a = model.coef_  # 回归系数
b = model.intercept_  # 截距
score = model.score(X, Y)  # R检验
# print(a,b,score)

# 预测车辆增量，2016-2049年，2050之后补上算
Car_increase_predict = a[0, 0] * GDP_speed + a[0, 1] * Population_speed + b
Car_increase_predict[0:6] = np.array([2752, 2813, 2673, 2578, 2424, 2622])
# print(Car_increase_predict)
########################################################



########################################################
# 以下为开启迭代部分
# 定义并初始化车辆总数2013-2050，单位（万）
Car_number = np.zeros(38)
Car_number[0:9] = np.array([13693, 15400, 17181, 19393, 21697, 23982, 26150, 28120, 30247])
# 定义并初始化电动车总数和比例2013-2050，单位（万）
# 注意电动车比例从2035年之后就变为100%，所以用np.ones
EV_number = np.zeros(38)
EV_number[0:9] = np.array([0, 22, 42, 91, 153, 266.74, 381, 492, 784])
# 采用三次样条插值补全车辆比例
EV_real = np.array([0, 22, 26, 49, 65, 113, 120, 187, 295]) / np.array([2017, 2188, 2385, 2752, 2813, 2673, 2578, 2424, 2622])
# 三次样条
model = interpolate.splrep(np.hstack([np.arange(2019, 2022), np.array(2035)]), np.hstack([EV_real[6:], np.array(1)]), k=3)
EV_predict = interpolate.splev(np.arange(2019, 2036), model)
EV_ratio = np.hstack([EV_real[0:6], EV_predict, np.ones(15)])
"""
plt.plot(np.arange(2013, 2051), EV_ratio)
plt.show()
"""
# 定义并初始化每年车代数量2013-2050
Car_general = np.zeros([38, 8])
Car_general[0] = np.array([0.033, 0.101, 0.115, 0.360, 0.391, 0, 0, 0]) * (Car_number[0] - EV_number[0])
Car_general[1] = np.array([0.021, 0.082, 0.093, 0.318, 0.452, 0.034, 0, 0]) * (Car_number[1] - EV_number[1])
Car_general[2] = np.array([0.014, 0.068, 0.077, 0.278, 0.501, 0.062, 0, 0]) * (Car_number[2] - EV_number[2])
Car_general[3] = np.array([0.010, 0.054, 0.064, 0.243, 0.524, 0.105, 0, 0]) * (Car_number[3] - EV_number[3])
Car_general[4] = np.array([0.001, 0.037, 0.055, 0.217, 0.475, 0.215, 0, 0]) * (Car_number[4] - EV_number[4])
Car_general[5] = np.array([0.000, 0.030, 0.045, 0.191, 0.425, 0.309, 0, 0]) * (Car_number[5] - EV_number[5])
# 定义每一个车代以及电动车的车龄占比，但用数量来计算方便，分布为国0-国6b共8个加上电动车
Car_ratio = np.zeros([9, 38, 30])
# 对车龄下面进行初始化，为0-29岁
# 对电动车有2013-2021的，其中电动车在Car_ratio[8]里面
# 对其他只有2016-2018的，前面后面是迭代出来的，直接读取得到
data = pd.read_csv('CarDistribution.csv')
for i in range(0, 9):
    for j in range(0, 9):
        Car_ratio[i, j, :] = ID = np.hstack([np.asarray(data.iloc[:, i * 9 + j]), np.zeros(6)])

# 原函数为1/(1+b*e**(c*Year))，对fx做变换，可以得到cx+lnb=ln(1/fx-1)，则对Year和右边的进行线性拟合即可
# 输入论文中的年龄分布数据
Year_fix = np.array([5, 7, 10, 12, 15, 17]).reshape(6, 1)
Car_ratio_2012 = np.array([508, 510, 289, 155, 31, 11]).reshape(6, 1) / 528
Car_ratio_2014 = np.array([524, 512, 416, 277, 70, 21]).reshape(6, 1) / 528
Car_ratio_2016 = np.array([524, 512, 437, 325, 114, 44]).reshape(6, 1) / 528
b = np.zeros(39)
c = np.zeros(39)
# 进行一元线性回归
# 2012
model = LinearRegression()
model.fit(Year_fix, np.log(1 / Car_ratio_2012 - 1))
b[0] = np.exp(model.intercept_)  # 截距
c[0] = model.coef_[0]  # 回归系数
# score = model.score(Year_fix, np.log(1 / Car_ratio_2012 - 1))  # R检验
# 2014
model = LinearRegression()
model.fit(Year_fix, np.log(1 / Car_ratio_2014 - 1))
b[2] = np.exp(model.intercept_)  # 截距
c[2] = model.coef_[0]  # 回归系数
# score = model.score(Year_fix, np.log(1 / Car_ratio_2014 - 1))  # R检验
# 2016
model = LinearRegression()
model.fit(Year_fix, np.log(1 / Car_ratio_2016 - 1))
b[4] = np.exp(model.intercept_)  # 截距
c[4] = model.coef_[0]  # 回归系数
# score = model.score(Year_fix, np.log(1 / Car_ratio_2016 - 1))  # R检验

# 对2013和2015则取中值
Car_ratio_2013 = (Car_ratio_2012 + Car_ratio_2014) / 2
Car_ratio_2015 = (Car_ratio_2014 + Car_ratio_2016) / 2
# 2013
model = LinearRegression()
model.fit(Year_fix, np.log(1 / Car_ratio_2013 - 1))
b[1] = np.exp(model.intercept_)  # 截距
c[1] = model.coef_[0]  # 回归系数
# score = model.score(Year_fix, np.log(1 / Car_ratio_2013 - 1))  # R检验
# 2015
model = LinearRegression()
model.fit(Year_fix, np.log(1 / Car_ratio_2015 - 1))
b[3] = np.exp(model.intercept_)  # 截距
c[3] = model.coef_[0]  # 回归系数
# score = model.score(Year_fix, np.log(1 / Car_ratio_2015 - 1))  # R检验

"""
# 输出
Year = np.arange(0, 30)
plt.plot(Year, 1 / (1 + b[0] * e ** (c[0] * Year)))
plt.plot(Year, 1 / (1 + b[1] * e ** (c[2] * Year)))
plt.plot(Year, 1 / (1 + b[2] * e ** (c[2] * Year)))
plt.plot(Year, 1 / (1 + b[3] * e ** (c[3] * Year)))
plt.plot(Year, 1 / (1 + b[4] * e ** (c[4] * Year)))
plt.show()
"""

# 对2016年之后的以从0到15年的0.8^i的速度进行平移，为2017-2050
Year = np.arange(0, 30)

Solve = np.array([0.9, 0.7, 0.5, 0.3, 0.1]).reshape(5, 1)
Year_fix = np.array([9.201635, 11.445915, 12.854565, 14.263215, 16.507495]).reshape(5, 1)
Distance = np.array([1.49788, 1.41214, 1.35831, 1.3045, 1.21875]).reshape(5, 1) / 2

for i in range(1, 35):
    Year_fix = Year_fix + Distance * (1 - 1 / (1 + e ** -(0.05 * i)))
    model = LinearRegression()
    model.fit(Year_fix, np.log(1 / Solve - 1))
    b[i + 4] = np.exp(model.intercept_)  # 截距
    c[i + 4] = model.coef_[0]  # 回归系数
    # score = model.score(Year_fix, np.log(1 / Car_ratio_2016 - 1))  # R检验
    # plt.plot(Year_fix, Car_ratio_2016, 'o', Year, 1 / (1 + b[i + 4] * e ** (c[i + 4] * Year)))

# 计算2013-2050的生存率，共38年，为1-29共29个长度，注意第一个值是2012要丢掉
Car_survival = np.zeros([38, 29])
for i in range(1, 39):
    Car_survival_now = 1 / (1 + b[i] * e ** (c[i] * Year))
    # 生存率为前除以后
    Car_survival[i - 1] = np.hstack([Car_survival_now[0], np.divide(Car_survival_now[1:-1], Car_survival_now[0:-2])])

"""
# 输出数据
for i in range(0, 39):
    Total = tuple(1 / (1 + b[i] * e ** (c[i] * Year)))
    # 直接变成DataFrame为一整列，需要转置变成一行
    df_T = pd.DataFrame(Total)
    df = pd.DataFrame(df_T.values.T)
    # 写入csv文件，mode = 'a'表示只在末尾追加写入，但是可能写入重复数据，注意最后导入的时候要进行查重清洗
    df.to_csv('1.csv', index=False, header=False, mode='a')
"""

"""
# 输出数据
Total = np.vstack([b, c])
# 直接变成DataFrame为一整列，需要转置变成一行
df_T = pd.DataFrame(Total)
df = pd.DataFrame(df_T.values.T)
# 写入csv文件，mode = 'a'表示只在末尾追加写入，但是可能写入重复数据，注意最后导入的时候要进行查重清洗
df.to_csv('1.csv', index=False, header=False, mode='a')
"""

"""
# 画图
for i in range(0, 38):
    plt.plot(np.arange(0, 31), 1 / (1 + b[i] * e ** (c[i] * np.arange(0, 31))))
plt.ylim(0,1)
plt.show()
"""

# 开始迭代，为2022-2050
for i in range(9, 38):
    # 首先对旧车进行报废操作
    for j in range(0, 9):
        Car_ratio[j, i, :] = np.hstack([np.array(0), Car_ratio[j, i - 1, :-1] * Car_survival[i - 1]])
    # 加入新车，Car_increase_predict从2016年开始的
    # 加入电动新车
    Car_ratio[8, i, 0] = Car_increase_predict[i - 3] * EV_ratio[i]
    # 对2022年，国六a占1/3，国六b占2/3，之后全部国六b
    if i == 2022:
        Car_ratio[7, i, 0] = Car_increase_predict[i - 3] * (1 - EV_ratio[i]) * 2 / 3
        Car_ratio[6, i, 0] = Car_increase_predict[i - 3] * (1 - EV_ratio[i]) / 3
    else:
        Car_ratio[7, i, 0] = Car_increase_predict[i - 3] * (1 - EV_ratio[i])

# 计算分类总数
Sum = np.zeros([9, 38])
for i in range(0, 38):
    for j in range(0, 9):
        Sum[j, i] = np.sum(Car_ratio[j, i])

# 计算总数
for i in range(0, 38):
    Car_number[i] = np.sum(Sum[:, i])

# 计算平均车龄
Car_age = np.zeros(38)
for i in range(0, 38):
    for j in range(0, 9):
        for k in range(0, 30):
            Car_age[i] = Car_age[i] + k * Car_ratio[j, i, k] / Car_number[i]

# 输出每种车龄的数量
Car_age_year = np.zeros([38, 30])
for i in range(0, 38):
    for j in range(0, 30):
        Car_age_year[i, j] = Car_age_year[i, j] + np.sum(Car_ratio[:, i, j])


# print(Car_age)
# print(Car_number)
# print(Car_number / np.hstack((Population[1:8] * 10, Population_new[:])) * 10)
# print(np.hstack((Population[1:8] * 10, Population_new[:])))

Total = Car_age_year[:, 0]
for i in range(1, 30):
    Total = np.vstack([Total, Car_age_year[:, i]])
# 直接变成DataFrame为一整列，需要转置变成一行
df_T = pd.DataFrame(Total)
df = pd.DataFrame(df_T.values.T)
# 写入csv文件，mode = 'a'表示只在末尾追加写入，但是可能写入重复数据，注意最后导入的时候要进行查重清洗
df.to_csv('1.csv', index=False, header=False, mode='a')


##############################################################


##############################################################
# 开始画图
# 指定颜色
Color = ['#000056', '#3537DE', '#7079DE', '#9168CE', '#D15B7E', '#FC6F68', '#FFB36A', '#FFDA43', '#63E5B3']
plt.rcParams['font.sans-serif'] = ['Arial']
# 计算占比
Ratio = np.zeros([9, 38])
for i in range(0, 38):
    Ratio[:, i] = Sum[:, i] / Car_number[i]


# 画图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.97, bottom=0.13, right=0.90, left=0.13, hspace=0, wspace=0)
# 去掉右侧和顶部边框
# axs.spines['top'].set_visible(False)
# 设置图片大小
fig.set_size_inches(4500 / 900, 3900 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.xlim(2013, 2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
axs.set_ylabel("Ratio of different emission standard (%)", fontsize=14)
axs.tick_params(labelsize=14, width=1, direction='in')   # 设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
# axs.spines['bottom'].set_linewidth(1)
# axs.spines['right'].set_linewidth(0.3)
# 输出图像
width_all = 0.9
axs.bar(np.arange(2013, 2051), Ratio[0]*100, color=Color[0], width=width_all, label='Lower limit')
axs.bar(np.arange(2013, 2051), Ratio[1]*100, bottom=(Ratio[0])*100, color=Color[1], width=width_all, label='Lower limit')
axs.bar(np.arange(2013, 2051), Ratio[2]*100, bottom=(Ratio[0] + Ratio[1])*100, color=Color[2], width=width_all, label='Lower limit')
axs.bar(np.arange(2013, 2051), Ratio[3]*100, bottom=(Ratio[0] + Ratio[1] + Ratio[2])*100, color=Color[3], width=width_all, label='Lower limit')
axs.bar(np.arange(2013, 2051), Ratio[4]*100, bottom=(Ratio[0] + Ratio[1] + Ratio[2] + Ratio[3])*100, color=Color[4], width=width_all, label='Lower limit')
axs.bar(np.arange(2013, 2051), Ratio[5]*100, bottom=(Ratio[4] + Ratio[0] + Ratio[1] + Ratio[2] + Ratio[3])*100, color=Color[5], width=width_all, label='Lower limit')
axs.bar(np.arange(2013, 2051), Ratio[6]*100, bottom=(Ratio[5] + Ratio[4] + Ratio[0] + Ratio[1] + Ratio[2] + Ratio[3])*100, color=Color[6], width=width_all, label='Lower limit')
axs.bar(np.arange(2013, 2051), Ratio[7]*100, bottom=(Ratio[6] + Ratio[5] + Ratio[4] + Ratio[0] + Ratio[1] + Ratio[2] + Ratio[3])*100, color=Color[7], width=width_all, label='Lower limit')
axs.bar(np.arange(2013, 2051), Ratio[8]*100, bottom=(Ratio[7] + Ratio[6] + Ratio[5] + Ratio[4] + Ratio[0] + Ratio[1] + Ratio[2] + Ratio[3])*100, color=Color[8], width=width_all, label='Lower limit')
# 设置坐标轴范围
axs.set_ylim((0, 100))
axs.set_yticks(np.arange(20, 100.1, 20))
# 第二个轴
ax2 = axs.twinx()
ax2.set_ylabel("Average vehicle age (year)", fontsize=14)
ax2.tick_params(labelsize=14, width=1, direction='in')   # 设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
# ax2.spines['bottom'].set_linewidth(1)
# ax2.spines['left'].set_linewidth(0.3)
# ax2.spines['top'].set_visible(False)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
ax2.plot(np.arange(2013, 2051), Car_age, color='k', linewidth=2)
# 设置坐标轴范围
ax2.set_ylim((0, 15))
ax2.set_yticks(np.arange(3, 15.1, 3))
# plt.legend(fontsize=12, loc='upper right')
# 保存
plt.savefig(r'D:\Fig0.png', dpi=900)
plt.close()



# GDP变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.15, right=0.95, left=0.1, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(6500 / 900, 3000 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.ylabel("GDP in PPP terms (US$ trillions)", fontsize=14)
plt.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
plt.plot(np.arange(2016, 2051), GDP_out[1:], color='#7079DE', linewidth=2)
# 设置坐标轴范围
plt.ylim((0, 60))
plt.xlim((2016, 2050))
# 设置坐标轴刻度
plt.yticks(np.array([15, 30, 45, 60]))
plt.xticks(np.array([2016, 2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=8, loc='upper right')
# 保存
plt.savefig(r'D:\Fig45.png', dpi=900)
plt.close()


# 人口变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.15, right=0.95, left=0.1, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(6500 / 900, 3000 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.ylabel("Population of China (10$^{9}$)", fontsize=14)
plt.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
plt.plot(np.arange(2016, 2051), Population_out[1:] / 1000000, color='#7079DE', linewidth=2)
# 设置坐标轴范围
plt.ylim((1.3, 1.5))
plt.xlim((2016, 2050))
# 设置坐标轴刻度
plt.yticks(np.array([1.35, 1.4, 1.45, 1.5]))
plt.xticks(np.array([2016, 2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=8, loc='upper right')
# 保存
plt.savefig(r'D:\Fig46.png', dpi=900)
plt.close()


# 新增车辆数
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.19, right=0.91, left=0.21, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(2800 / 900, 2100 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.ylabel("Vehicle sales (10$^{7}$)", fontsize=14)
plt.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
plt.bar(np.arange(2013, 2051), np.hstack([np.array([2017, 2188, 2385]), Car_increase_predict]) / 1000, color='#FFB64D')
# 设置坐标轴范围
plt.ylim((1.5, 3))
plt.xlim((2013, 2050))
# 设置坐标轴刻度
plt.yticks(np.array([1.8, 2.1, 2.4, 2.7, 3]))
plt.xticks(np.array([2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=8, loc='upper right')
# 保存
plt.savefig(r'D:\Fig1.png', dpi=900)
plt.close()


# 报废曲线图
Color = ['k', '#000089', '#3537DE', '#7079DE', '#9168CE', '#D15B7E', '#FC6F68', '#FFA06A', '#FFC36A', '#FFDA43', '#FFF89E']
# 分别存储对应颜色的16位和10位信息
# 7个颜色，分别对应2013，2019，2025，2031，2037，2043，2050，除了最后一个间隔7，其他为6
Color16 = ['#7079DE', '#D15B7E', '#FC6F68', '#FFA06A', '#FFB64D', '#FFDA43', '#FFE88E']
Color10 = np.array([[112, 121, 222], [209, 91, 126], [252, 111, 104], [255, 160, 106], [255, 182, 77], [255, 218, 67], [255, 232, 142]])
# 9个颜色，对应2013,2018,2023,2028,2033,2038,2043,2048,2050
# Color16 = ['#000000', '#00006f', '#3537DE', '#7079DE', '#9168CE', '#D15B7E', '#FC6F68', '#FFB36A', '#FFDA43']
# Color10 = np.array([[0, 0, 0], [0, 0, 111], [53, 55, 222], [112, 121, 222], [145, 104, 206], [209, 91, 126], [252, 111, 104], [255, 179, 106], [255, 218, 67]])
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.19, right=0.92, left=0.25, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(2500 / 900, 2100 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.ylabel("Survival rate (%)", fontsize=14)
plt.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
# 画图
for i in range(0, 38):
    # 计算当前的颜色
    flag = min(int(i / 6), 5)   # 用来指示颜色位置
    if i >= 32:  # 确定区间长度
        Len = 7
    else:
        Len = 6
    # 确定RGB
    Red = int(Color10[flag, 0] + (Color10[flag + 1, 0] - Color10[flag, 0]) * (i - 6 * flag) / Len)
    Green = int(Color10[flag, 1] + (Color10[flag + 1, 1] - Color10[flag, 1]) * (i - 6 * flag) / Len)
    Blue = int(Color10[flag, 2] + (Color10[flag + 1, 2] - Color10[flag, 2]) * (i - 6 * flag) / Len)
    # 转换为颜色代码
    Color_now = '#' + str(hex(Red)[2:]) + str(hex(Green)[2:]) + str(hex(Blue)[2:])
    print(Color_now)
    plt.plot(np.arange(0, 31, 0.1), 100 / (1 + b[i] * e ** (c[i] * np.arange(0, 31, 0.1))), color=Color_now, linewidth=0.4)
# 设置坐标轴范围
plt.ylim((0, 100))
plt.xlim((0, 31))
# 设置坐标轴刻度
plt.yticks(np.array([20, 40, 60, 80, 100]))
plt.xticks(np.array([0, 5, 10, 15, 20, 25, 30]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=8, loc='upper right')
# 保存
plt.savefig(r'D:\Fig2.png', dpi=900)
plt.close()


# EV发展图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.19, right=0.92, left=0.25, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(2500 / 900, 2100 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.ylabel("EV of vehicle sales (%)", fontsize=14)
plt.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
# 画图
plt.plot(np.arange(2013, 2051), EV_ratio * 100, color='#7079DE', linewidth=2)
plt.plot((2035,2035), (0,100), color='grey', linestyle='--', linewidth=2)
# 设置坐标轴范围
plt.ylim((0, 100))
plt.xlim((2013, 2050))
# 设置坐标轴刻度
plt.yticks(np.array([20, 40, 60, 80, 100]))
plt.xticks(np.array([2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=8, loc='upper right')
# 保存
plt.savefig(r'D:\Fig3.png', dpi=900)
plt.close()


# 车辆总数图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.19, right=0.92, left=0.25, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(2500 / 900, 2100 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.ylabel("Vehicle ownership (10$^{8}$)", fontsize=14)
plt.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
# 画图
plt.plot(np.arange(2013, 2051), Car_number / 10000, color='#FC6F68', linewidth=2)
# 设置坐标轴范围
plt.ylim((0, 6))
plt.xlim((2013, 2050))
# 设置坐标轴刻度
plt.yticks(np.array([1.2, 2.4, 3.6, 4.8, 6]))
plt.xticks(np.array([2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=8, loc='upper right')
# 保存
plt.savefig(r'D:\Fig47.png', dpi=900)
plt.close()