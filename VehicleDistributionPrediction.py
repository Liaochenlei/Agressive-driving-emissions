import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

####################################################
# The following is filling missing data
# Enter the number of vehicles from 2013 to 2050 (Unit: 10k)
Year = np.arange(2013, 2051)
Carnumber = np.array([13693,15400,17181,19393,21697,23982,25769,27628,29463,31804.94992,34044.9303,36156.71163,
                      38106.64283,39860.43801,41394.39986,42707.89362,43827.43163,44797.75473,45662.82892,46449.33236,
                      47164.08942,47805.65961,48377.43616,48892.76751,49371.42382,49831.12422,50280.61766,50718.82043,
                      51140.13879,51540.08822,51918.92196,52278.52627,52621.45758,52949.43377,53263.54309,53564.30982,
                      53852.103,54127.04318])

# Enter the proportion we forecast
S_max_real = np.array([4 / 25, 2 / 11, 12 / 49, 28 / 107])
S_min_real = np.array([12 / 25, 5 / 11, 17 / 49, 32 / 107])
S_mid_real = 1 - S_max_real - S_min_real

# Convert proportion to car number
Year_real = np.array([2013, 2014, 2018, 2021])
Car_max_real = S_max_real * np.array([Carnumber[0], Carnumber[1], Carnumber[5], Carnumber[8]])
Car_min_real = S_min_real * np.array([Carnumber[0], Carnumber[1], Carnumber[5], Carnumber[8]])
Car_mid_real = S_mid_real * np.array([Carnumber[0], Carnumber[1], Carnumber[5], Carnumber[8]])

# Perform cubic spline interpolation, filling the data from 2013 to 2021
model_max = interpolate.splrep(Year_real, Car_max_real, k=3)
model_min = interpolate.splrep(Year_real, Car_min_real, k=3)
model_mid = interpolate.splrep(Year_real, Car_mid_real, k=3)
Car_max_predict = interpolate.splev(np.arange(2013, 2022), model_max)
Car_min_predict = interpolate.splev(np.arange(2013, 2022), model_min)
Car_mid_predict = interpolate.splev(np.arange(2013, 2022), model_mid)

# plt the figure
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
# The following are the predictions made by ARIMA

# Prediction of three driver behaviors
model = ARIMA(Car_max_predict, order=(1, 1, 1))
model_fit = model.fit()
# Predict the number from 2000 to 2050
forecastdata = model_fit.forecast(29)
Car_max_ARIMA = np.hstack([Car_max_predict, forecastdata])

model = ARIMA(Car_min_predict, order=(1, 1, 1))
model_fit = model.fit()
# Predict the number from 2000 to 2050
forecastdata = model_fit.forecast(29)
Car_min_ARIMA = np.hstack([Car_min_predict, forecastdata])

model = ARIMA(Car_mid_predict, order=(1, 1, 1))
model_fit = model.fit()
# Predict the number from 2000 to 2050
forecastdata = model_fit.forecast(29)
Car_mid_ARIMA = np.hstack([Car_mid_predict, forecastdata])

plt.plot(Year_real, Car_max_real, 'o', Year, Car_max_ARIMA, color='r')
plt.plot(Year_real, Car_min_real, 'o', Year, Car_min_ARIMA, color='g')
plt.plot(Year_real, Car_mid_real, 'o', Year, Car_mid_ARIMA, color='b')
plt.show()

# Convert car number to proportion
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
####################################################

# Export Data
Total = np.vstack([Car_max_ratio * Carnumber, Car_mid_ratio * Carnumber, Car_min_ratio * Carnumber])
df_T = pd.DataFrame(Total)
df = pd.DataFrame(df_T.values.T)
# Write to csv
df.to_csv('VehicleDistribution.csv', index=False, header=False, mode='a')
