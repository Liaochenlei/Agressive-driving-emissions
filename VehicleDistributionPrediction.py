import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

####################################################
# 以下是年份补全
# 输入2013-2050的车龄数量（单位：w)
Year = np.arange(2013, 2051)
Carnumber = np.array([13693,15400,17181,19393,21697,23982,25769,27628,29463,31804.94992,34044.9303,36156.71163,
                      38106.64283,39860.43801,41394.39986,42707.89362,43827.43163,44797.75473,45662.82892,46449.33236,
                      47164.08942,47805.65961,48377.43616,48892.76751,49371.42382,49831.12422,50280.61766,50718.82043,
                      51140.13879,51540.08822,51918.92196,52278.52627,52621.45758,52949.43377,53263.54309,53564.30982,
                      53852.103,54127.04318])

# 输入我们预测的比例变化，分别为2013, 2014, 2018, 2021
S_max_real = np.array([4 / 25, 2 / 11, 12 / 49, 28 / 107])
S_min_real = np.array([12 / 25, 5 / 11, 17 / 49, 32 / 107])
S_mid_real = 1 - S_max_real - S_min_real

# 转化为数量
Year_real = np.array([2013, 2014, 2018, 2021])
Car_max_real = S_max_real * np.array([Carnumber[0], Carnumber[1], Carnumber[5], Carnumber[8]])
Car_min_real = S_min_real * np.array([Carnumber[0], Carnumber[1], Carnumber[5], Carnumber[8]])
Car_mid_real = S_mid_real * np.array([Carnumber[0], Carnumber[1], Carnumber[5], Carnumber[8]])

# 进行三次样条插值，补全2013-2021年之间的插值
model_max = interpolate.splrep(Year_real, Car_max_real, k=3)
model_min = interpolate.splrep(Year_real, Car_min_real, k=3)
model_mid = interpolate.splrep(Year_real, Car_mid_real, k=3)
Car_max_predict = interpolate.splev(np.arange(2013, 2022), model_max)
Car_min_predict = interpolate.splev(np.arange(2013, 2022), model_min)
Car_mid_predict = interpolate.splev(np.arange(2013, 2022), model_mid)


plt.plot(Year_real, Car_max_real, 'o', np.arange(2013, 2022), Car_max_predict, color='r')
plt.plot(Year_real, Car_min_real, 'o', np.arange(2013, 2022), Car_min_predict, color='g')
plt.plot(Year_real, Car_mid_real, 'o', np.arange(2013, 2022), Car_mid_predict, color='b')
plt.show()

plt.plot(Year_real, S_max_real, 'o', np.arange(2013, 2022), Car_max_predict / (Car_max_predict + Car_min_predict + Car_mid_predict), color='r')
plt.plot(Year_real, S_min_real, 'o', np.arange(2013, 2022), Car_min_predict / (Car_max_predict + Car_min_predict + Car_mid_predict), color='g')
plt.plot(Year_real, S_mid_real, 'o', np.arange(2013, 2022), Car_mid_predict / (Car_max_predict + Car_min_predict + Car_mid_predict), color='b')
plt.show()
####################################################


####################################################
# 以下是ARIMA进行预测

# 分别对三种预测
model = ARIMA(Car_max_predict, order=(1, 1, 1))
model_fit = model.fit()
# 进行预测，为2022-2050
forecastdata = model_fit.forecast(29)
Car_max_ARIMA = np.hstack([Car_max_predict, forecastdata])

model = ARIMA(Car_min_predict, order=(1, 1, 1))
model_fit = model.fit()
# 进行预测，为2022-2050
forecastdata = model_fit.forecast(29)
Car_min_ARIMA = np.hstack([Car_min_predict, forecastdata])

model = ARIMA(Car_mid_predict, order=(1, 1, 1))
model_fit = model.fit()
# 进行预测，为2022-2050
forecastdata = model_fit.forecast(29)
Car_mid_ARIMA = np.hstack([Car_mid_predict, forecastdata])

plt.plot(Year_real, Car_max_real, 'o', Year, Car_max_ARIMA, color='r')
plt.plot(Year_real, Car_min_real, 'o', Year, Car_min_ARIMA, color='g')
plt.plot(Year_real, Car_mid_real, 'o', Year, Car_mid_ARIMA, color='b')
plt.show()

# 折算为比例
Sum = Car_max_ARIMA + Car_mid_ARIMA + Car_min_ARIMA
Car_max_ratio = Car_max_ARIMA / Sum
Car_mid_ratio = Car_mid_ARIMA / Sum
Car_min_ratio = Car_min_ARIMA / Sum
plt.plot(Year_real, S_max_real, 'o', Year, Car_max_ratio, color='r', label='max')
plt.plot(Year_real, S_mid_real, 'o', Year, Car_mid_ratio, color='g', label='mid')
plt.plot(Year_real, S_min_real, 'o', Year, Car_min_ratio, color='b', label='min')
plt.ylim(0, 0.6)
# plt.legend()
plt.show()


# 读出数据
Total = np.vstack([Car_max_ratio * Carnumber, Car_mid_ratio * Carnumber, Car_min_ratio * Carnumber])
# 直接变成DataFrame为一整列，需要转置变成一行
df_T = pd.DataFrame(Total)
df = pd.DataFrame(df_T.values.T)
# 写入csv文件，mode = 'a'表示只在末尾追加写入，但是可能写入重复数据，注意最后导入的时候要进行查重清洗
df.to_csv('VehicleDistribution.csv', index=False, header=False, mode='a')
