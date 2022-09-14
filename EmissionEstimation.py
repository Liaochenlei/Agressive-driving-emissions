import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from statsmodels.tsa.arima.model import ARIMA

# 读入数据
data = pd.read_csv('VehicleDistribution.csv')
year = np.asarray(data.iloc[:, 0])
car_number = np.asarray(data.iloc[:, 1])
Totalcar = np.asarray(data.iloc[:, 2])
Rate_CO = np.asarray(data.iloc[:, 3])
Rate_HC = np.asarray(data.iloc[:, 4])
Rate_NOx = np.asarray(data.iloc[:, 5])
Rate_PM = np.asarray(data.iloc[:, 6])
car_max = np.asarray(data.iloc[:, 7])
car_min = np.asarray(data.iloc[:, 8])
car_mid = np.asarray(data.iloc[:, 9])
S_rate_max = car_max / Totalcar
S_rate_min = car_min / Totalcar
S_rate_mid = 1 - S_rate_min - S_rate_max
Highway_c = np.asarray(data.iloc[:, 10])
City_c = np.asarray(data.iloc[:, 11])


plt.rcParams['font.sans-serif'] = ['Arial']  # 指定默认字体，有Arial和Times New Roman
Color = ['#7079DE', '#FC6F68', '#FFB64D']
Dis = ['Calm', 'Mediate', 'Aggressive']
place = np.array([0, 1, 5, 8])
year_real = [2013, 2014, 2018, 2021]

# 计算拥堵次数，由75%改为95%置信区间
Center = 96  # 系数中位数
Low = 75    # 系数最低点
Up = 111     # 系数最高点
Per = 5  # 饱和流占总拥堵的比例

Times_high = Highway_c * Center * Per * 60 * 24 / Totalcar / 10000
Times_city = City_c * Center * Per * 60 * 4 / Totalcar / 10000
Times = Times_high + Times_city



CO2_min_real = np.array([9.979958639, 9.798465262, 9.305623263, 9.120657266])
CO2_mid_real = np.array([13.3559041, 13.296321552, 13.04369774, 12.90740446])
CO2_max_real = np.array([22.41830104, 22.105321042, 21.50748082, 21.37702338])


CO_max_real = np.array([0.154924535, 0.149532052, 0.12647445, 0.112111419])
CO_mid_real = np.array([0.088246325, 0.084320621, 0.064810356, 0.060626141])
CO_min_real = np.array([0.061557338, 0.058324562, 0.046563313, 0.043784535])


HC_max_real = np.array([0.003978637, 0.003725672, 0.002780762, 0.002226574])
HC_mid_real = np.array([0.002686075, 0.002403517, 0.001943152, 0.0015637])
HC_min_real = np.array([0.002234845, 0.002056125, 0.001536337, 0.001236799])

NOx_max_real = np.array([0.014191308, 0.012867212, 0.00916753, 0.008086254])
NOx_mid_real = np.array([0.009931148, 0.009035210, 0.006288869, 0.00541641])
NOx_min_real = np.array([0.008309782, 0.007923105, 0.005481023, 0.00473535])

# PM10与PM2.5是固定的1.13倍的关系，只需要计算10乘以1+1/1.13即为两者的和
PM_max_real = np.array([0.00059342, 0.00054232, 0.000343192, 0.000237823])
PM_mid_real = np.array([0.000338145, 0.00030952, 0.000144958, 9.47843E-05])
PM_min_real = np.array([0.000225671, 0.000204321, 9.72861E-05, 6.76638E-05])


"""
#输出污染参数
Total = np.vstack((CO2_max_real, CO2_mid_real, CO2_min_real, CO_max_real, CO_mid_real, CO_min_real, HC_max_real, HC_mid_real, HC_min_real, NOx_max_real, NOx_mid_real, NOx_min_real, PM_max_real, PM_mid_real, PM_min_real))
#直接变成DataFrame为一整列，需要转置变成一行
df_T = pd.DataFrame(Total)
df = pd.DataFrame(df_T.values.T)
#写入csv文件，mode = 'a'表示只在末尾追加写入，但是可能写入重复数据，注意最后导入的时候要进行查重清洗
df.to_csv('Pollution_real.csv', index = False, header = False, mode = 'a')
"""



# 输入参数
Year = np.arange(2013, 2051)
Year_real = np.array([2013, 2014, 2018, 2021])
Year_fix = np.arange(2013, 2022)

# 先进行三次样条补全
model_max = interpolate.splrep(Year_real, CO2_max_real, k=3)
model_min = interpolate.splrep(Year_real, CO2_min_real, k=3)
model_mid = interpolate.splrep(Year_real, CO2_mid_real, k=3)
CO2_max_predict = interpolate.splev(np.arange(2013, 2022), model_max)
CO2_mid_predict = interpolate.splev(np.arange(2013, 2022), model_mid)
CO2_min_predict = interpolate.splev(np.arange(2013, 2022), model_min)

# ARIMA预测
model = ARIMA(CO2_max_predict, order=(1, 1, 1))
model_fit = model.fit()
# 进行预测，为2022-2050
forecastdata = model_fit.forecast(29)
CO2_max = np.hstack([CO2_max_predict, forecastdata])

model = ARIMA(CO2_min_predict, order=(1, 1, 1))
model_fit = model.fit()
# 进行预测，为2022-2050
forecastdata = model_fit.forecast(29)
CO2_min = np.hstack([CO2_min_predict, forecastdata])

model = ARIMA(CO2_mid_predict, order=(1, 1, 1))
model_fit = model.fit()
# 进行预测，为2022-2050
forecastdata = model_fit.forecast(29)
CO2_mid = np.hstack([CO2_mid_predict, forecastdata])

#计算多出来的CO2
CO2_year = (CO2_max * S_rate_max + CO2_mid * S_rate_mid + CO2_min * S_rate_min - CO2_min) * Times * 365 * car_number / 100
CO2 = np.zeros(len(CO2_year))
CO2[0] = CO2_year[0]
for i in range(1, len(CO2_year)):
    CO2[i] = CO2[i - 1] + CO2_year[i]


#计算其他污染物
# CO
CO_max = np.zeros(len(year))
CO_mid = np.zeros(len(year))
CO_min = np.zeros(len(year))
CO_max[place] = CO_max_real
CO_mid[place] = CO_mid_real
CO_min[place] = CO_min_real

for i in range(2,5):
    CO_max[i] = CO_max[1] - (CO_max[1] - CO_max[1] * Rate_CO[i] / Rate_CO[1]) / (CO_max[1] - CO_max[1] * Rate_CO[5] / Rate_CO[1]) * (CO_max[1] - CO_max[5])
    CO_mid[i] = CO_mid[1] - (CO_mid[1] - CO_mid[1] * Rate_CO[i] / Rate_CO[1]) / (CO_mid[1] - CO_mid[1] * Rate_CO[5] / Rate_CO[1]) * (CO_mid[1] - CO_mid[5])
    CO_min[i] = CO_min[1] - (CO_min[1] - CO_min[1] * Rate_CO[i] / Rate_CO[1]) / (CO_min[1] - CO_min[1] * Rate_CO[5] / Rate_CO[1]) * (CO_min[1] - CO_min[5])

for i in range(6,8):
    CO_max[i] = CO_max[5] - (CO_max[5] - CO_max[5] * Rate_CO[i] / Rate_CO[5]) / (CO_max[5] - CO_max[5] * Rate_CO[8] / Rate_CO[5]) * (CO_max[5] - CO_max[8])
    CO_mid[i] = CO_mid[5] - (CO_mid[5] - CO_mid[5] * Rate_CO[i] / Rate_CO[5]) / (CO_mid[5] - CO_mid[5] * Rate_CO[8] / Rate_CO[5]) * (CO_mid[5] - CO_mid[8])
    CO_min[i] = CO_min[5] - (CO_min[5] - CO_min[5] * Rate_CO[i] / Rate_CO[5]) / (CO_min[5] - CO_min[5] * Rate_CO[8] / Rate_CO[5]) * (CO_min[5] - CO_min[8])

for i in range(9,len(year)):
    CO_max[i] = CO_max[8] * (Rate_CO[8] - (Rate_CO[8] - Rate_CO[i])) / Rate_CO[8]
    CO_mid[i] = CO_mid[8] * (Rate_CO[8] - (Rate_CO[8] - Rate_CO[i])) / Rate_CO[8]
    CO_min[i] = CO_min[8] * (Rate_CO[8] - (Rate_CO[8] - Rate_CO[i])) / Rate_CO[8]

CO2_max[place] = CO2_max_real
CO2_mid[place] = CO2_mid_real
CO2_min[place] = CO2_min_real

# HC
HC_max = np.zeros(len(year))
HC_mid = np.zeros(len(year))
HC_min = np.zeros(len(year))
HC_max[place] = HC_max_real
HC_mid[place] = HC_mid_real
HC_min[place] = HC_min_real

for i in range(2,5):
    HC_max[i] = HC_max[1] - (HC_max[1] - HC_max[1] * Rate_HC[i] / Rate_HC[1]) / (HC_max[1] - HC_max[1] * Rate_HC[5] / Rate_HC[1]) * (HC_max[1] - HC_max[5])
    HC_mid[i] = HC_mid[1] - (HC_mid[1] - HC_mid[1] * Rate_HC[i] / Rate_HC[1]) / (HC_mid[1] - HC_mid[1] * Rate_HC[5] / Rate_HC[1]) * (HC_mid[1] - HC_mid[5])
    HC_min[i] = HC_min[1] - (HC_min[1] - HC_min[1] * Rate_HC[i] / Rate_HC[1]) / (HC_min[1] - HC_min[1] * Rate_HC[5] / Rate_HC[1]) * (HC_min[1] - HC_min[5])

for i in range(6,8):
    HC_max[i] = HC_max[5] - (HC_max[5] - HC_max[5] * Rate_HC[i] / Rate_HC[5]) / (HC_max[5] - HC_max[5] * Rate_HC[8] / Rate_HC[5]) * (HC_max[5] - HC_max[8])
    HC_mid[i] = HC_mid[5] - (HC_mid[5] - HC_mid[5] * Rate_HC[i] / Rate_HC[5]) / (HC_mid[5] - HC_mid[5] * Rate_HC[8] / Rate_HC[5]) * (HC_mid[5] - HC_mid[8])
    HC_min[i] = HC_min[5] - (HC_min[5] - HC_min[5] * Rate_HC[i] / Rate_HC[5]) / (HC_min[5] - HC_min[5] * Rate_HC[8] / Rate_HC[5]) * (HC_min[5] - HC_min[8])

for i in range(9,len(year)):
    HC_max[i] = HC_max[8] * (Rate_HC[8] - (Rate_HC[8] - Rate_HC[i])) / Rate_HC[8]
    HC_mid[i] = HC_mid[8] * (Rate_HC[8] - (Rate_HC[8] - Rate_HC[i])) / Rate_HC[8]
    HC_min[i] = HC_min[8] * (Rate_HC[8] - (Rate_HC[8] - Rate_HC[i])) / Rate_HC[8]


# NOx
NOx_max = np.zeros(len(year))
NOx_mid = np.zeros(len(year))
NOx_min = np.zeros(len(year))
NOx_max[place] = NOx_max_real
NOx_mid[place] = NOx_mid_real
NOx_min[place] = NOx_min_real

for i in range(2,5):
    NOx_max[i] = NOx_max[1] - (NOx_max[1] - NOx_max[1] * Rate_NOx[i] / Rate_NOx[1]) / (NOx_max[1] - NOx_max[1] * Rate_NOx[5] / Rate_NOx[1]) * (NOx_max[1] - NOx_max[5])
    NOx_mid[i] = NOx_mid[1] - (NOx_mid[1] - NOx_mid[1] * Rate_NOx[i] / Rate_NOx[1]) / (NOx_mid[1] - NOx_mid[1] * Rate_NOx[5] / Rate_NOx[1]) * (NOx_mid[1] - NOx_mid[5])
    NOx_min[i] = NOx_min[1] - (NOx_min[1] - NOx_min[1] * Rate_NOx[i] / Rate_NOx[1]) / (NOx_min[1] - NOx_min[1] * Rate_NOx[5] / Rate_NOx[1]) * (NOx_min[1] - NOx_min[5])

for i in range(6,8):
    NOx_max[i] = NOx_max[5] - (NOx_max[5] - NOx_max[5] * Rate_NOx[i] / Rate_NOx[5]) / (NOx_max[5] - NOx_max[5] * Rate_NOx[8] / Rate_NOx[5]) * (NOx_max[5] - NOx_max[8])
    NOx_mid[i] = NOx_mid[5] - (NOx_mid[5] - NOx_mid[5] * Rate_NOx[i] / Rate_NOx[5]) / (NOx_mid[5] - NOx_mid[5] * Rate_NOx[8] / Rate_NOx[5]) * (NOx_mid[5] - NOx_mid[8])
    NOx_min[i] = NOx_min[5] - (NOx_min[5] - NOx_min[5] * Rate_NOx[i] / Rate_NOx[5]) / (NOx_min[5] - NOx_min[5] * Rate_NOx[8] / Rate_NOx[5]) * (NOx_min[5] - NOx_min[8])

for i in range(9,len(year)):
    NOx_max[i] = NOx_max[8] * (Rate_NOx[8] - (Rate_NOx[8] - Rate_NOx[i])) / Rate_NOx[8]
    NOx_mid[i] = NOx_mid[8] * (Rate_NOx[8] - (Rate_NOx[8] - Rate_NOx[i])) / Rate_NOx[8]
    NOx_min[i] = NOx_min[8] * (Rate_NOx[8] - (Rate_NOx[8] - Rate_NOx[i])) / Rate_NOx[8]


# PM
PM_max = np.zeros(len(year))
PM_mid = np.zeros(len(year))
PM_min = np.zeros(len(year))
PM_max[place] = PM_max_real
PM_mid[place] = PM_mid_real
PM_min[place] = PM_min_real

for i in range(2,5):
    PM_max[i] = PM_max[1] - (PM_max[1] - PM_max[1] * Rate_PM[i] / Rate_PM[1]) / (PM_max[1] - PM_max[1] * Rate_PM[5] / Rate_PM[1]) * (PM_max[1] - PM_max[5])
    PM_mid[i] = PM_mid[1] - (PM_mid[1] - PM_mid[1] * Rate_PM[i] / Rate_PM[1]) / (PM_mid[1] - PM_mid[1] * Rate_PM[5] / Rate_PM[1]) * (PM_mid[1] - PM_mid[5])
    PM_min[i] = PM_min[1] - (PM_min[1] - PM_min[1] * Rate_PM[i] / Rate_PM[1]) / (PM_min[1] - PM_min[1] * Rate_PM[5] / Rate_PM[1]) * (PM_min[1] - PM_min[5])

for i in range(6,8):
    PM_max[i] = PM_max[5] - (PM_max[5] - PM_max[5] * Rate_PM[i] / Rate_PM[5]) / (PM_max[5] - PM_max[5] * Rate_PM[8] / Rate_PM[5]) * (PM_max[5] - PM_max[8])
    PM_mid[i] = PM_mid[5] - (PM_mid[5] - PM_mid[5] * Rate_PM[i] / Rate_PM[5]) / (PM_mid[5] - PM_mid[5] * Rate_PM[8] / Rate_PM[5]) * (PM_mid[5] - PM_mid[8])
    PM_min[i] = PM_min[5] - (PM_min[5] - PM_min[5] * Rate_PM[i] / Rate_PM[5]) / (PM_min[5] - PM_min[5] * Rate_PM[8] / Rate_PM[5]) * (PM_min[5] - PM_min[8])

for i in range(9,len(year)):
    PM_max[i] = PM_max[8] * (Rate_PM[8] - (Rate_PM[8] - Rate_PM[i])) / Rate_PM[8]
    PM_mid[i] = PM_mid[8] * (Rate_PM[8] - (Rate_PM[8] - Rate_PM[i])) / Rate_PM[8]
    PM_min[i] = PM_min[8] * (Rate_PM[8] - (Rate_PM[8] - Rate_PM[i])) / Rate_PM[8]



#计算污染变化
CO_year = (CO_max * S_rate_max + CO_mid * S_rate_mid + CO_min * S_rate_min - CO_min) * Times * 365 * car_number / 100
HC_year = (HC_max * S_rate_max + HC_mid * S_rate_mid + HC_min * S_rate_min - HC_min) * Times * 365 * car_number / 100
NOx_year = (NOx_max * S_rate_max + NOx_mid * S_rate_mid + NOx_min * S_rate_min - NOx_min) * Times * 365 * car_number / 100
PM_year = (PM_max * S_rate_max + PM_mid * S_rate_mid + PM_min * S_rate_min - PM_min) * Times * 365 * car_number / 100

Total_pol_year = CO_year + HC_year + NOx_year + PM_year
Total_pol = np.zeros(len(year))
Total_pol[0] = Total_pol_year[0]
for i in range(1, len(year)):
    Total_pol[i] = Total_pol[i - 1] + Total_pol_year[i]

CO_all = np.zeros(len(year))
CO_all[0] = CO_year[0]
for i in range(1, len(year)):
    CO_all[i] = CO_all[i - 1] + CO_year[i]

HC_all = np.zeros(len(year))
HC_all[0] = HC_year[0]
for i in range(1, len(year)):
    HC_all[i] = HC_all[i - 1] + HC_year[i]

NOx_all = np.zeros(len(year))
NOx_all[0] = NOx_year[0]
for i in range(1, len(year)):
    NOx_all[i] = NOx_all[i - 1] + NOx_year[i]

PM_all = np.zeros(len(year))
PM_all[0] = PM_year[0]
for i in range(1, len(year)):
    PM_all[i] = PM_all[i - 1] + PM_year[i]


"""
#输出污染参数
Total = np.vstack((CO_min, CO_mid, CO_max, HC_min, HC_mid, HC_max, NOx_min, NOx_mid, NOx_max, PM_min, PM_mid, PM_max, CO2_min, CO2_mid, CO2_max,))
#直接变成DataFrame为一整列，需要转置变成一行
df_T = pd.DataFrame(Total)
df = pd.DataFrame(df_T.values.T)
#写入csv文件，mode = 'a'表示只在末尾追加写入，但是可能写入重复数据，注意最后导入的时候要进行查重清洗
df.to_csv('Pollution.csv', index = False, header = False, mode = 'a')


#输出CO2总共与单车参数
mid = (CO2_max * S_rate_max + CO2_mid * S_rate_mid + CO2_min * S_rate_min)
Total = np.vstack((CO2_year, CO2, CO2_min * Times * 365 / 10 ** 3, mid * Times * 365 / 10 ** 3, CO2_max * Times * 365 / 10 ** 3, (mid - CO2_min) / (CO2_max - CO2_min)))
#直接变成DataFrame为一整列，需要转置变成一行
df_T = pd.DataFrame(Total)
df = pd.DataFrame(df_T.values.T)
#写入csv文件，mode = 'a'表示只在末尾追加写入，但是可能写入重复数据，注意最后导入的时候要进行查重清洗
df.to_csv('CO2.csv', index = False, header = False, mode = 'a')


#输出污染物总共与单车参数
CO_use_mid = (CO_max * S_rate_max + CO_mid * S_rate_mid + CO_min * S_rate_min) * Times * 365
HC_use_mid = (HC_max * S_rate_max + HC_mid * S_rate_mid + HC_min * S_rate_min) * Times * 365
NOx_use_mid = (NOx_max * S_rate_max + NOx_mid * S_rate_mid + NOx_min * S_rate_min) * Times * 365
PM_use_mid = (PM_max * S_rate_max + PM_mid * S_rate_mid + PM_min * S_rate_min) * Times * 365
Total_mid = CO_use_mid + HC_use_mid + NOx_use_mid + PM_use_mid
Total_min = (CO_min + HC_min + NOx_min + PM_min) * Times * 365
Total_max = (CO_max + HC_max + NOx_max + PM_max) * Times * 365

Total = np.vstack((Total_pol_year, Total_pol, Total_min / 1000, Total_mid / 1000, Total_max / 1000, (Total_mid - Total_min) / (Total_max - Total_min)))
#直接变成DataFrame为一整列，需要转置变成一行
df_T = pd.DataFrame(Total)
df = pd.DataFrame(df_T.values.T)
#写入csv文件，mode = 'a'表示只在末尾追加写入，但是可能写入重复数据，注意最后导入的时候要进行查重清洗
df.to_csv('Pollutant.csv', index = False, header = False, mode = 'a')
"""





# S比例变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.18, right=0.95, left=0.15, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(3800 / 900, 2500 / 900)

# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.ylabel("Driver type proportion (%)", fontsize=14)
plt.tick_params(labelsize=14, width=0.3, direction='in')   # 设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
plt.plot(year, S_rate_min * 100, color=Color[0], label=Dis[0])
plt.plot(year, S_rate_mid * 100, color=Color[1], label=Dis[1])
plt.plot(year, S_rate_max * 100, color=Color[2], label=Dis[2])
# plt.scatter(year_real, S_min_real, color=Color[0])
# plt.scatter(year_real, S_mid_real, color=Color[1])
# plt.scatter(year_real, S_max_real, color=Color[2])
# 设置坐标轴范围
plt.ylim((0, 80))
plt.xlim((2013, 2015))
# 设置坐标轴刻度
plt.yticks(np.array([20, 40, 60, 80]))
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=6, loc='upper right')
# 保存
plt.savefig(r'D:\Fig10.png', dpi=900)
plt.close()




# S总量变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.18, right=0.95, left=0.15, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(3800 / 900, 2500 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.ylabel("Driver numbers (10$^{8}$)", fontsize=14)
plt.tick_params(labelsize=14, width=0.3, direction='in')   # 设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
plt.plot(year, car_min / 10000, color=Color[0], label=Dis[0])
plt.plot(year, car_mid / 10000, color=Color[1], label=Dis[1])
plt.plot(year, car_max / 10000, color=Color[2], label=Dis[2])
plt.plot(year, Totalcar / 10000, color='k', linestyle='--', label='Total')
# plt.fill_between(year, 0, car_min, color='C0')
# plt.fill_between(year, car_min, car_mid + car_min, color='C1')
# plt.fill_between(year, car_mid + car_min, car_max + car_mid + car_min, color='C2')
# plt.scatter(year_real, car_min_real)
# plt.scatter(year_real, Totalcar[place] - car_min_real - car_max_real)
# plt.scatter(year_real, car_max_real)
# 设置坐标轴范围
plt.ylim((0, 8))
plt.xlim((2013, 2015))
#设置坐标轴刻度
plt.yticks(np.array([2, 4, 6, 8]))
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=6, loc='upper right')
# 保存
plt.savefig(r'D:\Fig11.png', dpi=900)
plt.close()



# CO2变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.18, right=0.90, left=0.2, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(2500 / 900, 2500 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.ylabel("CO$_{2}$ emission (g)", fontsize=14)
plt.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
plt.plot(year, CO2_min, color=Color[0], label=Dis[0])
plt.plot(year, CO2_mid, color=Color[1], label=Dis[1])
plt.plot(year, CO2_max, color=Color[2], label=Dis[2])
#设置坐标轴范围
plt.ylim((0, 36))
plt.xlim((2013, 2050))
#设置坐标轴刻度
plt.yticks(np.array([9, 18, 27, 36]))
plt.xticks(np.array([2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
#plt.legend(fontsize=8, loc='upper right')
# 保存
plt.savefig(r'D:\Fig12.png', dpi=900)
plt.close()



"""
# 其他污染变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.18, right=0.90, left=0.18, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(3500 / 900, 2500 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.ylabel("Total pollutants (10$^{-1}$ g)", fontsize=14)
plt.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
plt.plot(year, (CO_min + PM_min + NOx_min + HC_min) * 10, color=Color[0], label=Dis[0])
plt.plot(year, (CO_mid + PM_mid + NOx_mid + HC_mid) * 10, color=Color[1], label=Dis[1])
plt.plot(year, (CO_max + PM_max + NOx_max + HC_max) * 10, color=Color[2], label=Dis[2])
#设置坐标轴范围
plt.ylim((0, 2))
plt.xlim((2013, 2050))
#设置坐标轴刻度
plt.yticks(np.array([0.5, 1.0, 1.5, 2.0]))
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
#plt.legend(fontsize=8, loc='upper right')
# 保存
plt.savefig(r'D:\Fig13.png', dpi=900)
plt.close()
"""



# CO变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.18, right=0.90, left=0.22, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(2500 / 900, 2500 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.ylabel("CO emission (10$^{-1}$ g)", fontsize=14)
plt.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
plt.plot(year, CO_min * 10, color=Color[0], label=Dis[0])
plt.plot(year, CO_mid * 10, color=Color[1], label=Dis[1])
plt.plot(year, CO_max * 10, color=Color[2], label=Dis[2])
# 设置坐标轴范围
plt.ylim((0, 2))
plt.xlim((2013, 2050))
plt.xticks(np.array([2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=8, loc='upper right')
# 设置坐标轴刻度
plt.yticks(np.array([0.5, 1, 1.5, 2]))
# 保存
plt.savefig(r'D:\Fig14.png', dpi=900)
plt.close()


# HC变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.18, right=0.90, left=0.22, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(2500 / 900, 2500 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.ylabel("HC emission (10$^{-3}$ g)", fontsize=14)
plt.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
plt.plot(year, HC_min * 1000, color=Color[0], label=Dis[0])
plt.plot(year, HC_mid * 1000, color=Color[1], label=Dis[1])
plt.plot(year, HC_max * 1000, color=Color[2], label=Dis[2])
# 设置坐标轴范围
plt.ylim((0, 6))
plt.xlim((2013, 2050))
plt.xticks(np.array([2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
#plt.legend(fontsize=8, loc='upper right')
# 设置坐标轴刻度
plt.yticks(np.array([1.5, 3.0, 4.5, 6.0]))
# 保存
plt.savefig(r'D:\Fig15.png', dpi=900)
plt.close()



# NOx变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.18, right=0.90, left=0.22, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(2500 / 900, 2500 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.ylabel("NO$_{x}$ emission (10$^{-2}$ g)", fontsize=14)
plt.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
plt.plot(year, NOx_min * 100, color=Color[0], label=Dis[0])
plt.plot(year, NOx_mid * 100, color=Color[1], label=Dis[1])
plt.plot(year, NOx_max * 100, color=Color[2], label=Dis[2])
#设置坐标轴范围
plt.ylim((0, 2))
plt.xlim((2013, 2050))
plt.xticks(np.array([2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
#plt.legend(fontsize=8, loc='upper right')
#设置坐标轴刻度
plt.yticks(np.array([0.5, 1.0, 1.5, 2.0]))
# 保存
plt.savefig(r'D:\Fig16.png', dpi=900)
plt.close()


# PM变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.18, right=0.90, left=0.22, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(2500 / 900, 2500 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.ylabel("PM emission (10$^{-4}$ g)", fontsize=14)
plt.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
plt.plot(year, PM_min * 10000, color=Color[0], label=Dis[0])
plt.plot(year, PM_mid * 10000, color=Color[1], label=Dis[1])
plt.plot(year, PM_max * 10000, color=Color[2], label=Dis[2])
#设置坐标轴范围
plt.ylim((0, 6.4))
plt.xlim((2013, 2050))
plt.xticks(np.array([2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
#plt.legend(fontsize=8, loc='upper right')
#设置坐标轴刻度
plt.yticks(np.array([1.6, 3.2, 4.8, 6.4]))
# 保存
plt.savefig(r'D:\Fig17.png', dpi=900)
plt.close()



"""
# CO2总量变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
# 设置图片大小
fig.set_size_inches(5500 / 900, 3000 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
axs.set_ylabel("CO$_{2}$ emission (10$^{7}$ ton/year)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
bwith = 0.1
#plt.bar(year, CO2_year / 10 ** 7, yerr=CO2_year/10/ 10 ** 7, color='C0')
axs.bar(year, CO2_year / 10 ** 7 * Low / Center, color=Color[0], width=1, alpha=0.8, label='Lower bound', edgecolor='black', linewidth=0.3)
axs.bar(year, CO2_year / 10 ** 7 - CO2_year / 10 ** 7 * Low / Center, color=Color[1], width=1, alpha=0.8, bottom=CO2_year / 10 ** 7 * Low / Center , label='Predicted value', edgecolor='black', linewidth=0.3)
axs.bar(year, CO2_year / 10 ** 7 * Up / Center - CO2_year / 10 ** 7, color=Color[2], width=1, alpha=0.8, bottom=CO2_year / 10 ** 7, label='Upper bound', edgecolor='black', linewidth=0.3)
#axs.plot(year, CO2_year / 10 ** 7 * Low / Center, color=Color[0], alpha=1, label='Lower limit', linewidth=1)
#axs.plot(year, CO2_year / 10 ** 7, color=Color[1], alpha=1, label='Predictive value', linewidth=1)
#axs.plot(year, CO2_year / 10 ** 7 * Up / Center, color=Color[2], alpha=1, label='Upper limit', linewidth=1)
#axs.fill_between(year, np.zeros(len(year)), CO2_year / 10 ** 7 * Low / Center, color=Color[0], alpha = 0.4, edgecolor='None')
#axs.fill_between(year, CO2_year / 10 ** 7 * Low / Center, CO2_year / 10 ** 7, color=Color[1], alpha = 0.4, edgecolor='None')
#axs.fill_between(year, CO2_year / 10 ** 7, CO2_year / 10 ** 7 * Up / Center, color=Color[2], alpha = 0.4, edgecolor='None')
#设置坐标轴范围
plt.legend(fontsize=14, loc='upper left')
axs.set_ylim((0, 5))
axs.set_yticks(np.arange(1,6))
#第二个轴
ax2 = axs.twinx()
ax2.spines['top'].set_visible(False)
ax2.set_ylabel("Accumulated CO$_{2}$ emission (10$^{8}$ ton)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
ax2.spines['bottom'].set_linewidth(0.3)
ax2.spines['right'].set_linewidth(0.3)
# 输出图像
ax2.plot(year, CO2 / 10 ** 8, color='k', linewidth=1)
ax2.fill_between(year, CO2 / 10 ** 8 * Low / Center, CO2 / 10 ** 8 * Up / Center, color='grey', alpha = 0.4, edgecolor='None')
#设置坐标轴范围
ax2.set_ylim((0, 10))
ax2.set_yticks(np.arange(2, 11, 2))
# 保存
plt.savefig(r'D:\Fig18.png', dpi=900)
plt.close()
"""

"""
# 其他污染总量变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
# 设置图片大小
fig.set_size_inches(5500 / 900, 3000 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
axs.set_ylabel("Total pollution (10$^{5}$ ton/year)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
axs.bar(year, Total_pol_year / 10 ** 4 * Low / Center / 10, color=Color[0], width=1, alpha=0.8, label='Lower bound', edgecolor='black', linewidth=0.3)
axs.bar(year, Total_pol_year / 10 ** 4 / 10- Total_pol_year / 10 ** 4 * Low / Center / 10, color=Color[1], width=1, alpha=0.8, bottom=Total_pol_year / 10 ** 4 * Low / Center / 10, label='Predicted value', edgecolor='black', linewidth=0.3)
axs.bar(year, Total_pol_year / 10 ** 4 * Up / Center / 10 - Total_pol_year / 10 ** 4 / 10, color=Color[2], width=1, alpha=0.8, bottom=Total_pol_year / 10 ** 4 / 10, label='Upper bound', edgecolor='black', linewidth=0.3)
#设置坐标轴范围
plt.legend(fontsize=14, loc='upper left')
axs.set_ylim((0, 2))
axs.set_yticks(np.arange(0.4,2.1,0.4))
#第二个轴
ax2 = axs.twinx()
ax2.spines['top'].set_visible(False)
ax2.set_ylabel("Accumulated pollution (10$^{6}$ ton)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
ax2.spines['bottom'].set_linewidth(0.3)
ax2.spines['right'].set_linewidth(0.3)
# 输出图像
ax2.plot(year, Total_pol / 10 ** 5 / 10, color='k', linewidth=1)
ax2.fill_between(year, Total_pol / 10 ** 5 * Low / Center / 10, Total_pol / 10 ** 5 * Up / Center / 10, color='grey', alpha = 0.4, edgecolor='None')
#设置坐标轴范围
ax2.set_ylim((0, 4))
ax2.set_yticks(np.arange(0.8,4.1,0.8))
# 保存
plt.savefig(r'D:\Fig19.png', dpi=900)
plt.close()
"""

"""
# 单车全年CO2总量变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# 设置图片大小
fig.set_size_inches(5500 / 900, 3000 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
axs.spines['top'].set_visible(False)
axs.set_ylabel("Wasted proportion", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(1)
axs.spines['right'].set_linewidth(0.3)
# 输出图像
mid = (CO2_max * S_rate_max + CO2_mid * S_rate_mid + CO2_min * S_rate_min)
#axs.plot(year, (mid - CO2_min) / (CO2_max - CO2_min), color='black', linewidth=1, alpha=0.5)
#axs.fill_between(year, np.zeros(len(year)), (mid - CO2_min) / (CO2_max - CO2_min), color='grey', alpha = 0.2, edgecolor='None')
axs.bar(year, (mid - CO2_min) / (CO2_max - CO2_min), color='gray', width=1, alpha=0.2, label='Lower limit', edgecolor='black')
#设置坐标轴范围
axs.set_ylim((0.2, 0.7))
axs.set_yticks(np.arange(0.3, 0.71, 0.1))

#第二个轴
ax2 = axs.twinx()
# 去掉右侧和顶部边框
ax2.spines['top'].set_visible(False)
ax2.set_ylabel("CO$_{2}$ emission (kg/veh·year)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
ax2.plot(year, CO2_min * Times * 365 / 10 ** 3, color=Color[0], linewidth=3, label=Dis[0])
ax2.plot(year, mid * Times * 365 / 10 ** 3, color=Color[1], linewidth=3, label='Averaged emission')
ax2.plot(year, CO2_max * Times * 365 / 10 ** 3, color=Color[2], linewidth=3, label=Dis[2])
#设置坐标轴范围
ax2.set_ylim((0, 300))
ax2.set_yticks(np.arange(60, 301, 60))
plt.legend(fontsize=12, loc='upper right')
# 保存
plt.savefig(r'D:\Fig22.png', dpi=900)
plt.close()
"""


CO_use_mid = (CO_max * S_rate_max + CO_mid * S_rate_mid + CO_min * S_rate_min) * Times * 365
HC_use_mid = (HC_max * S_rate_max + HC_mid * S_rate_mid + HC_min * S_rate_min) * Times * 365
NOx_use_mid = (NOx_max * S_rate_max + NOx_mid * S_rate_mid + NOx_min * S_rate_min) * Times * 365
PM_use_mid = (PM_max * S_rate_max + PM_mid * S_rate_mid + PM_min * S_rate_min) * Times * 365


Total_mid = CO_use_mid + HC_use_mid + NOx_use_mid + PM_use_mid
Total_min = (CO_min + HC_min + NOx_min + PM_min) * Times * 365
Total_max = (CO_max + HC_max + NOx_max + PM_max) * Times * 365

"""
# 单车全年污染总量变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
# 设置图片大小
fig.set_size_inches(5500 / 900, 3000 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
axs.set_ylabel("Wasted proportion", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(1)
axs.spines['right'].set_linewidth(0.3)
# 输出图像
axs.bar(year, (Total_mid - Total_min) / (Total_max - Total_min), color='gray', width=1, alpha=0.2, label='Lower limit', edgecolor='black')
#axs.plot(year, (Total_mid - Total_min) / (Total_max - Total_min), color='black', linewidth=1, alpha=0.5)
#axs.fill_between(year, np.zeros(len(year)), (Total_mid - Total_min) / (Total_max - Total_min), color='grey', alpha = 0.2, edgecolor='None')
#设置坐标轴范围
axs.set_ylim((0.2, 0.7))
axs.set_yticks(np.arange(0.3, 0.71, 0.1))
#第二个轴
ax2 = axs.twinx()
ax2.set_ylabel("Total pollution (kg/veh·year)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['left'].set_linewidth(0.3)
ax2.spines['top'].set_visible(False)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
ax2.plot(year, Total_min / 1000, color=Color[0], linewidth=3, label=Dis[0])
ax2.plot(year, Total_mid / 1000, color=Color[1], linewidth=3, label='Averaged pollution')
ax2.plot(year, Total_max / 1000, color=Color[2], linewidth=3, label=Dis[2])
#设置坐标轴范围
ax2.set_ylim((0, 1.5))
ax2.set_yticks(np.arange(0.3, 1.8, 0.3))
plt.legend(fontsize=12, loc='upper right')
# 保存
plt.savefig(r'D:\Fig23.png', dpi=900)
plt.close()
"""


"""
# PM变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.18, right=0.90, left=0.18, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# 设置图片大小
fig.set_size_inches(3500 / 900, 2500 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.ylabel("PM emission (10$^{-4}$ g)", fontsize=14)
plt.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
plt.plot(year, PM_min * 10000, color=Color[0], label=Dis[0])
plt.plot(year, PM_mid * 10000, color=Color[1], label=Dis[1])
plt.plot(year, PM_max * 10000, color=Color[2], label=Dis[2])
#设置坐标轴范围
plt.ylim((0, 2.4))
plt.xlim((2010, 2050))
plt.xticks(np.array([2010, 2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
#plt.legend(fontsize=8, loc='upper right')
#设置坐标轴刻度
plt.yticks(np.array([0.6, 1.2, 1.8, 2.4]))
# 保存
plt.savefig(r'D:\Fig31.png', dpi=900)
plt.close()

"""


# CO总量变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
# 设置图片大小
fig.set_size_inches(4000 / 900, 3000 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
axs.set_ylabel("CO$_{2}$ emission (10$^{7}$ ton/year)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
bwith = 0.1
axs.bar(year, CO2_year / 10 ** 7 * Low / Center, color=Color[0], width=1, alpha=0.8, label='Lower bound', edgecolor='black', linewidth=0.3)
axs.bar(year, CO2_year / 10 ** 7 - CO2_year / 10 ** 7 * Low / Center, color=Color[1], width=1, alpha=0.8, bottom=CO2_year / 10 ** 7 * Low / Center , label='Predicted value', edgecolor='black', linewidth=0.3)
axs.bar(year, CO2_year / 10 ** 7 * Up / Center - CO2_year / 10 ** 7, color=Color[2], width=1, alpha=0.8, bottom=CO2_year / 10 ** 7, label='Upper bound', edgecolor='black', linewidth=0.3)
#设置坐标轴范围
plt.legend(fontsize=14, loc='upper left')
axs.set_ylim((0, 5))
axs.set_yticks(np.arange(1,6))
#第二个轴
ax2 = axs.twinx()
ax2.spines['top'].set_visible(False)
ax2.set_ylabel("Accumulated CO$_{2}$ emission (10$^{8}$ ton)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
ax2.spines['bottom'].set_linewidth(0.3)
ax2.spines['right'].set_linewidth(0.3)
# 输出图像
ax2.plot(year, CO2 / 10 ** 8, color='k', linewidth=1)
ax2.fill_between(year, CO2 / 10 ** 8 * Low / Center, CO2 / 10 ** 8 * Up / Center, color='grey', alpha = 0.4, edgecolor='None')
#设置坐标轴范围
ax2.set_ylim((0, 10))
ax2.set_yticks(np.arange(2, 11, 2))
# 保存
plt.savefig(r'D:\Fig33.png', dpi=900)
plt.close()


# 单车全年CO2总量变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# 设置图片大小
fig.set_size_inches(4000 / 900, 3000 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
axs.spines['top'].set_visible(False)
axs.set_ylabel("Preventable CO$_{2}$ emission (%)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(1)
axs.spines['right'].set_linewidth(0.3)
# 输出图像
mid = (CO2_max * S_rate_max + CO2_mid * S_rate_mid + CO2_min * S_rate_min)
#axs.plot(year, (mid - CO2_min) / (CO2_max - CO2_min), color='black', linewidth=1, alpha=0.5)
#axs.fill_between(year, np.zeros(len(year)), (mid - CO2_min) / (CO2_max - CO2_min), color='grey', alpha = 0.2, edgecolor='None')
axs.bar(year, (mid - CO2_min) / (CO2_max - CO2_min) * 100, color='gray', width=1, alpha=0.2, label='Lower limit', edgecolor='black')
#设置坐标轴范围
axs.set_ylim((20, 70))
axs.set_yticks(np.arange(30, 71, 10))
#第二个轴
ax2 = axs.twinx()
# 去掉右侧和顶部边框
ax2.spines['top'].set_visible(False)
ax2.set_ylabel("CO$_{2}$ emission (kg/veh·year)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
ax2.plot(year, CO2_min * Times * 365 / 10 ** 3, color=Color[0], linewidth=3, label=Dis[0])
ax2.plot(year, mid * Times * 365 / 10 ** 3, color=Color[1], linewidth=3, label='Predicted')
ax2.plot(year, CO2_max * Times * 365 / 10 ** 3, color=Color[2], linewidth=3, label=Dis[2])
#设置坐标轴范围
ax2.set_ylim((0, 350))
ax2.set_yticks(np.arange(70, 351, 70))
plt.legend(fontsize=14, loc='upper right')
# 保存
plt.savefig(r'D:\Fig34.png', dpi=900)
plt.close()


CO_use_mid = (CO_max * S_rate_max + CO_mid * S_rate_mid + CO_min * S_rate_min) * Times * 365
HC_use_mid = (HC_max * S_rate_max + HC_mid * S_rate_mid + HC_min * S_rate_min) * Times * 365
NOx_use_mid = (NOx_max * S_rate_max + NOx_mid * S_rate_mid + NOx_min * S_rate_min) * Times * 365
PM_use_mid = (PM_max * S_rate_max + PM_mid * S_rate_mid + PM_min * S_rate_min) * Times * 365


# CO污染总量变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
# 设置图片大小
fig.set_size_inches(4000 / 900, 3000 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
axs.set_ylabel("CO emission (10$^{5}$ ton/year)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
axs.bar(year, CO_year / 10 ** 5 * Low / Center, color=Color[0], width=1, alpha=0.8, label='Lower bound', edgecolor='black', linewidth=0.3)
axs.bar(year, CO_year / 10 ** 5 - CO_year / 10 ** 5 * Low / Center, color=Color[1], width=1, alpha=0.8, bottom=CO_year / 10 ** 5 * Low / Center, label='Predicted value', edgecolor='black', linewidth=0.3)
axs.bar(year, CO_year / 10 ** 5 * Up / Center - CO_year / 10 ** 5, color=Color[2], width=1, alpha=0.8, bottom=CO_year / 10 ** 5, label='Upper bound', edgecolor='black', linewidth=0.3)
#设置坐标轴范围
plt.legend(fontsize=14, loc='upper left')
axs.set_ylim((0, 2))
axs.set_yticks(np.arange(0.4,2.1,0.4))
#第二个轴
ax2 = axs.twinx()
ax2.spines['top'].set_visible(False)
ax2.set_ylabel("Accumulated CO emission (10$^{6}$ ton)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
ax2.spines['bottom'].set_linewidth(0.3)
ax2.spines['right'].set_linewidth(0.3)
# 输出图像
ax2.plot(year, CO_all / 10 ** 6, color='k', linewidth=1)
ax2.fill_between(year, CO_all / 10 ** 6 * Low / Center, CO_all / 10 ** 6 * Up / Center, color='grey', alpha = 0.4, edgecolor='None')
#设置坐标轴范围
ax2.set_ylim((0, 3.5))
ax2.set_yticks(np.arange(0.7,3.6,0.7))
# 保存
plt.savefig(r'D:\Fig35.png', dpi=900)
plt.close()




# CO全年污染总量变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
# 设置图片大小
fig.set_size_inches(4000 / 900, 3000 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
axs.set_ylabel("Preventable CO emission (%)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(1)
axs.spines['right'].set_linewidth(0.3)
# 输出图像
axs.bar(year, (CO_use_mid - CO_min * Times * 365) / (CO_max * Times * 365 - CO_min * Times * 365) * 100, color='gray', width=1, alpha=0.2, label='Lower limit', edgecolor='black')
#axs.plot(year, (Total_mid - Total_min) / (Total_max - Total_min), color='black', linewidth=1, alpha=0.5)
#axs.fill_between(year, np.zeros(len(year)), (Total_mid - Total_min) / (Total_max - Total_min), color='grey', alpha = 0.2, edgecolor='None')
#设置坐标轴范围
axs.set_ylim((20, 70))
axs.set_yticks(np.arange(30, 71, 10))
#第二个轴
ax2 = axs.twinx()
ax2.set_ylabel("CO emission (kg/veh·year)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['left'].set_linewidth(0.3)
ax2.spines['top'].set_visible(False)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
ax2.plot(year, CO_min * Times * 365 / 1000, color=Color[0], linewidth=3, label=Dis[0])
ax2.plot(year, CO_use_mid / 1000, color=Color[1], linewidth=3, label='Predicted')
ax2.plot(year, CO_max * Times * 365 / 1000, color=Color[2], linewidth=3, label=Dis[2])
#设置坐标轴范围
ax2.set_ylim((0, 1.5))
ax2.set_yticks(np.arange(0.3, 1.6, 0.3))
plt.legend(fontsize=14, loc='upper right')
# 保存
plt.savefig(r'D:\Fig36.png', dpi=900)
plt.close()




# HC污染总量变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
# 设置图片大小
fig.set_size_inches(4000 / 900, 3000 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
axs.set_ylabel("HC emission (10$^{3}$ ton/year)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
axs.bar(year, HC_year / 10 ** 3 * Low / Center, color=Color[0], width=1, alpha=0.8, label='Lower bound', edgecolor='black', linewidth=0.3)
axs.bar(year, HC_year / 10 ** 3 - HC_year / 10 ** 3 * Low / Center, color=Color[1], width=1, alpha=0.8, bottom=HC_year / 10 ** 3 * Low / Center, label='Predicted value', edgecolor='black', linewidth=0.3)
axs.bar(year, HC_year / 10 ** 3 * Up / Center - HC_year / 10 ** 3, color=Color[2], width=1, alpha=0.8, bottom=HC_year / 10 ** 3, label='Upper bound', edgecolor='black', linewidth=0.3)
#设置坐标轴范围
plt.legend(fontsize=14, loc='upper left')
axs.set_ylim((0, 3))
axs.set_yticks(np.arange(0.6,3.1,0.6))
#第二个轴
ax2 = axs.twinx()
ax2.spines['top'].set_visible(False)
ax2.set_ylabel("Accumulated HC emission (10$^{4}$ ton)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
ax2.spines['bottom'].set_linewidth(0.3)
ax2.spines['right'].set_linewidth(0.3)
# 输出图像
ax2.plot(year, HC_all / 10 ** 4, color='k', linewidth=1)
ax2.fill_between(year, HC_all / 10 ** 4 * Low / Center, HC_all / 10 ** 4 * Up / Center, color='grey', alpha = 0.4, edgecolor='None')
#设置坐标轴范围
ax2.set_ylim((0, 3.5))
ax2.set_yticks(np.arange(0.9,4.6,0.9))
# 保存
plt.savefig(r'D:\Fig37.png', dpi=900)
plt.close()


# HC全年污染总量变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
# 设置图片大小
fig.set_size_inches(4000 / 900, 3000 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
axs.set_ylabel("Preventable HC emission (%)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(1)
axs.spines['right'].set_linewidth(0.3)
# 输出图像
axs.bar(year, (HC_use_mid - HC_min * Times * 365) / (HC_max * Times * 365 - HC_min * Times * 365) * 100, color='gray', width=1, alpha=0.2, label='Lower limit', edgecolor='black')
#设置坐标轴范围
axs.set_ylim((20, 70))
axs.set_yticks(np.arange(30, 71, 10))
#第二个轴
ax2 = axs.twinx()
ax2.set_ylabel("HC emission (10$^{-2}$kg/veh·year)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['left'].set_linewidth(0.3)
ax2.spines['top'].set_visible(False)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
ax2.plot(year, HC_min * Times * 365 / 10, color=Color[0], linewidth=3, label=Dis[0])
ax2.plot(year, HC_use_mid / 10, color=Color[1], linewidth=3, label='Predicted')
ax2.plot(year, HC_max * Times * 365 / 10, color=Color[2], linewidth=3, label=Dis[2])
#设置坐标轴范围
ax2.set_ylim((0, 3))
ax2.set_yticks(np.arange(0.6, 3.1, 0.6))
plt.legend(fontsize=14, loc='upper right')
# 保存
plt.savefig(r'D:\Fig38.png', dpi=900)
plt.close()


# PM污染总量变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
# 设置图片大小
fig.set_size_inches(4000 / 900, 3000 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
axs.set_ylabel("PM emission (10$^{2}$ ton/year)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
axs.bar(year, PM_year / 10 ** 2 * Low / Center, color=Color[0], width=1, alpha=0.8, label='Lower bound', edgecolor='black', linewidth=0.3)
axs.bar(year, PM_year / 10 ** 2 - PM_year / 10 ** 2 * Low / Center, color=Color[1], width=1, alpha=0.8, bottom=PM_year / 10 ** 2 * Low / Center, label='Predicted value', edgecolor='black', linewidth=0.3)
axs.bar(year, PM_year / 10 ** 2 * Up / Center - PM_year / 10 ** 2, color=Color[2], width=1, alpha=0.8, bottom=PM_year / 10 ** 2, label='Upper bound', edgecolor='black', linewidth=0.3)
#设置坐标轴范围
plt.legend(fontsize=14, loc='upper left')
axs.set_ylim((0, 5))
axs.set_yticks(np.arange(1,5.1,1))
#第二个轴
ax2 = axs.twinx()
ax2.spines['top'].set_visible(False)
ax2.set_ylabel("Accumulated PM emission (10$^{3}$ ton)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
ax2.spines['bottom'].set_linewidth(0.3)
ax2.spines['right'].set_linewidth(0.3)
# 输出图像
ax2.plot(year, PM_all / 10 ** 3, color='k', linewidth=1)
ax2.fill_between(year, PM_all / 10 ** 3 * Low / Center, PM_all / 10 ** 3 * Up / Center, color='grey', alpha = 0.4, edgecolor='None')
#设置坐标轴范围
ax2.set_ylim((0, 7))
ax2.set_yticks(np.arange(1.4,7.1,1.4))
# 保存
plt.savefig(r'D:\Fig41.png', dpi=900)
plt.close()



# PM全年污染总量变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
# 设置图片大小
fig.set_size_inches(4000 / 900, 3000 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
axs.set_ylabel("Preventable PM emission (%)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(1)
axs.spines['right'].set_linewidth(0.3)
# 输出图像
axs.bar(year, (PM_use_mid - PM_min * Times * 365) / (PM_max * Times * 365 - PM_min * Times * 365) * 100, color='gray', width=1, alpha=0.2, label='Lower limit', edgecolor='black')
#设置坐标轴范围
axs.set_ylim((20, 70))
axs.set_yticks(np.arange(30, 71, 10))
#第二个轴
ax2 = axs.twinx()
ax2.set_ylabel("PM emission (10$^{-3}$kg/veh·year)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['left'].set_linewidth(0.3)
ax2.spines['top'].set_visible(False)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
ax2.plot(year, PM_min * Times * 365, color=Color[0], linewidth=3, label=Dis[0])
ax2.plot(year, PM_use_mid, color=Color[1], linewidth=3, label='Predicted')
ax2.plot(year, PM_max * Times * 365, color=Color[2], linewidth=3, label=Dis[2])
#设置坐标轴范围
ax2.set_ylim((0, 3))
ax2.set_yticks(np.arange(0.6, 3.1, 0.6))
plt.legend(fontsize=14, loc='upper right')
# 保存
plt.savefig(r'D:\Fig42.png', dpi=900)
plt.close()


# NOx污染总量变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
# 设置图片大小
fig.set_size_inches(4000 / 900, 3000 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.xlim(2013, 2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
axs.set_ylabel("NO$_{x}$ emission (10$^{3}$ ton/year)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
axs.bar(year, NOx_year / 10 ** 3 * Low / Center, color=Color[0], width=1, alpha=0.8, label='Lower bound', edgecolor='black', linewidth=0.3)
axs.bar(year, NOx_year / 10 ** 3 - NOx_year / 10 ** 3 * Low / Center, color=Color[1], width=1, alpha=0.8, bottom=NOx_year / 10 ** 3 * Low / Center, label='Predicted value', edgecolor='black', linewidth=0.3)
axs.bar(year, NOx_year / 10 ** 3 * Up / Center - NOx_year / 10 ** 3, color=Color[2], width=1, alpha=0.8, bottom=NOx_year / 10 ** 3, label='Upper bound', edgecolor='black', linewidth=0.3)
#设置坐标轴范围
plt.legend(fontsize=14, loc='upper left')
axs.set_ylim((0, 8))
axs.set_yticks(np.arange(1.6, 8.1, 1.6))
#第二个轴
ax2 = axs.twinx()
ax2.spines['top'].set_visible(False)
ax2.set_ylabel("Accumulated NO$_{x}$ emission (10$^{5}$ ton)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
ax2.spines['bottom'].set_linewidth(0.3)
ax2.spines['right'].set_linewidth(0.3)
# 输出图像
ax2.plot(year, NOx_all / 10 ** 5, color='k', linewidth=1)
ax2.fill_between(year, NOx_all / 10 ** 5 * Low / Center, NOx_all / 10 ** 5 * Up / Center, color='grey', alpha = 0.4, edgecolor='None')
#设置坐标轴范围
ax2.set_ylim((0, 1.5))
ax2.set_yticks(np.arange(0.3, 1.6, 0.3))
# 保存
plt.savefig(r'D:\Fig43.png', dpi=900)
plt.close()



# Nox全年污染总量变化图
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# 去掉右侧和顶部边框
axs.spines['top'].set_visible(False)
# 设置图片大小
fig.set_size_inches(4000 / 900, 3000 / 900)
# 设置坐标轴
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
#plt.xticks(rotation=20)
axs.set_ylabel("Preventable NO$_{x}$ emission (%)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
axs.spines['bottom'].set_linewidth(1)
axs.spines['right'].set_linewidth(0.3)
# 输出图像
axs.bar(year, (NOx_use_mid - NOx_min * Times * 365) / (NOx_max * Times * 365 - NOx_min * Times * 365) * 100, color='gray', width=1, alpha=0.2, label='Lower limit', edgecolor='black')
#设置坐标轴范围
axs.set_ylim((20, 70))
axs.set_yticks(np.arange(30, 71, 10))
#第二个轴
ax2 = axs.twinx()
ax2.set_ylabel("NO$_{x}$ emission (10$^{-2}$kg/veh·year)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')   #设置坐标刻度值的字体大小和刻度粗细,刻度向内
# 修改底部坐标轴（x轴）的粗细
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['left'].set_linewidth(0.3)
ax2.spines['top'].set_visible(False)
# 仅开启y轴方向的坐标轴
plt.grid(axis='y', linewidth=0.2)
# 输出图像
ax2.plot(year, NOx_min * Times * 365 / 10, color=Color[0], linewidth=3, label=Dis[0])
ax2.plot(year, NOx_use_mid / 10, color=Color[1], linewidth=3, label='Predicted')
ax2.plot(year, NOx_max * Times * 365 / 10, color=Color[2], linewidth=3, label=Dis[2])
#设置坐标轴范围
ax2.set_ylim((0, 10))
ax2.set_yticks(np.arange(2, 10.1, 2))
plt.legend(fontsize=14, loc='upper right')
# 保存
plt.savefig(r'D:\Fig44.png', dpi=900)
plt.close()

"""
############################################
# 读出所有数据

# S比例变化图
Total = np.vstack([S_rate_min, S_rate_mid, S_rate_max])

# 单次变化图
# CO2变化图（单位g）
Total = np.vstack([Total, CO2_min, CO2_mid, CO2_max])
# CO变化图（单位10-1g）
Total = np.vstack([Total, CO_min * 10, CO_mid * 10, CO_max * 10])
# HC变化图（单位10-3g）
Total = np.vstack([Total, HC_min * 1000, HC_mid * 1000, HC_max * 1000])
# NOx变化图（单位10-2g）
Total = np.vstack([Total, NOx_min * 100, NOx_mid * 100, NOx_max * 100])
# PM变化图（单位10-4g）
Total = np.vstack([Total, PM_min * 10000, PM_mid * 10000, PM_max * 10000])


# 总量和去年变化图
# CO2逐年变化图（单位107ton/year）
Total = np.vstack([Total, CO2_year / 10 ** 7 * Low / Center, CO2_year / 10 ** 7, CO2_year / 10 ** 7 * Up / Center])
# CO2逐年变化图总量（单位108ton）
Total = np.vstack([Total, CO2 / 10 ** 8 * Low / Center, CO2 / 10 ** 8, CO2 / 10 ** 8 * Up / Center])
# 单车全年CO2总量变化图(kg/veh·year)
Total = np.vstack([Total, CO2_min * Times * 365 / 10 ** 3, mid * Times * 365 / 10 ** 3, CO2_max * Times * 365 / 10 ** 3])
# CO2比例
Total = np.vstack([Total, (mid - CO2_min) / (CO2_max - CO2_min) * 100])

# CO污染逐年（单位105ton/year）
Total = np.vstack([Total, CO_year / 10 ** 5 * Low / Center, CO_year / 10 ** 5, CO_year / 10 ** 5 * Up / Center])
# CO污染总量（单位106ton）
Total = np.vstack([Total, CO_all / 10 ** 6 * Low / Center, CO_all / 10 ** 6, CO_all / 10 ** 6 * Up / Center])
# 单车全年CO总量变化图(kg/veh·year)
Total = np.vstack([Total, CO_min * Times * 365 / 1000, CO_use_mid / 1000, CO_max * Times * 365 / 1000])
# CO比例
Total = np.vstack([Total, (CO_use_mid - CO_min * Times * 365) / (CO_max * Times * 365 - CO_min * Times * 365) * 100])

# HC污染总量变化图
# HC污染逐年（单位103ton/year）
Total = np.vstack([Total, HC_year / 10 ** 3 * Low / Center, HC_year / 10 ** 3, HC_year / 10 ** 3 * Up / Center])
# HC污染总量（单位104ton）
Total = np.vstack([Total, HC_all / 10 ** 4 * Low / Center, HC_all / 10 ** 4, HC_all / 10 ** 4 * Up / Center])
# 单车全年HC总量变化图(10-2kg/veh·year)
Total = np.vstack([Total, HC_min * Times * 365 / 10, HC_use_mid / 10, HC_max * Times * 365 / 10])
# HC比例
Total = np.vstack([Total, (HC_use_mid - HC_min * Times * 365) / (HC_max * Times * 365 - HC_min * Times * 365) * 100])

# PM污染总量变化图
# PM污染逐年（单位102ton/year）
Total = np.vstack([Total, PM_year / 10 ** 2 * Low / Center, PM_year / 10 ** 2, PM_year / 10 ** 2 * Up / Center])
# PM污染总量（单位103ton）
Total = np.vstack([Total, PM_all / 10 ** 3 * Low / Center, PM_all / 10 ** 3, PM_all / 10 ** 3 * Up / Center])
# 单车全年PM总量变化图(10-3kg/veh·year)
Total = np.vstack([Total, PM_min * Times * 365, PM_use_mid, PM_max * Times * 365])
# PM比例
Total = np.vstack([Total, (PM_use_mid - PM_min * Times * 365) / (PM_max * Times * 365 - PM_min * Times * 365) * 100])

# NOx污染总量变化图
# NOx污染逐年（单位103ton/year）
Total = np.vstack([Total, NOx_year / 10 ** 3 * Low / Center, NOx_year / 10 ** 3, NOx_year / 10 ** 3 * Up / Center])
# NOx污染总量（单位105ton）
Total = np.vstack([Total, NOx_all / 10 ** 5 * Low / Center, NOx_all / 10 ** 5, NOx_all / 10 ** 5 * Up / Center])
# 单车全年NOx总量变化图(10-2kg/veh·year)
Total = np.vstack([Total, NOx_min * Times * 365 / 10, NOx_use_mid / 10, NOx_max * Times * 365 / 10])
# NOx比例
Total = np.vstack([Total, (NOx_use_mid - NOx_min * Times * 365) / (NOx_max * Times * 365 - NOx_min * Times * 365) * 100])

# 直接变成DataFrame为一整列，需要转置变成一行
df_T = pd.DataFrame(Total)
df = pd.DataFrame(df_T.values.T)
# 写入csv文件，mode = 'a'表示只在末尾追加写入，但是可能写入重复数据，注意最后导入的时候要进行查重清洗
df.to_csv('1.csv', index=False, header=False, mode='a')
"""