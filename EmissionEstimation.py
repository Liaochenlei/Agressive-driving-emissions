import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from statsmodels.tsa.arima.model import ARIMA

# Read data
data0 = pd.read_csv('VehicleDistributionNumber.csv')
year = np.asarray(data0.iloc[:, 0])
car_number = np.asarray(data0.iloc[:, 1])
Totalcar = np.asarray(data0.iloc[:, 2])
car_max = np.asarray(data0.iloc[:, 3])
car_min = np.asarray(data0.iloc[:, 4])
car_mid = np.asarray(data0.iloc[:, 5])

data1 = pd.read_csv('ReductionFactor.csv')
Rate_CO = np.asarray(data1.iloc[:, 1])
Rate_HC = np.asarray(data1.iloc[:, 2])
Rate_NOx = np.asarray(data1.iloc[:, 3])
Rate_PM = np.asarray(data1.iloc[:, 4])

data2 = pd.read_csv('RoadCongestionMileage.csv')
Highway_c = np.asarray(data2.iloc[:, 1])
City_c = np.asarray(data2.iloc[:, 2])

S_rate_max = car_max / Totalcar
S_rate_min = car_min / Totalcar
S_rate_mid = 1 - S_rate_min - S_rate_max


plt.rcParams['font.sans-serif'] = ['Arial']     # Set default font
Color = ['#7079DE', '#FC6F68', '#FFB64D']   # Set Color
Dis = ['Calm', 'Mediate', 'Aggressive']     # Set Label

place = np.array([0, 1, 5, 8])
year_real = [2013, 2014, 2018, 2021]

# Calculate the number of congestion
# 95% Confidence Interval
Center = 96  # Median
Low = 75    # Minimum
Up = 111     # Maximum
Per = 5  # Ratio of saturated flow

Times_high = Highway_c * Center * Per * 60 * 24 / Totalcar / 10000
Times_city = City_c * Center * Per * 60 * 4 / Totalcar / 10000
Times = Times_high + Times_city

# Emissions
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

PM_max_real = np.array([0.00059342, 0.00054232, 0.000343192, 0.000237823])
PM_mid_real = np.array([0.000338145, 0.00030952, 0.000144958, 9.47843E-05])
PM_min_real = np.array([0.000225671, 0.000204321, 9.72861E-05, 6.76638E-05])


# Enter Year
Year = np.arange(2013, 2051)
Year_real = np.array([2013, 2014, 2018, 2021])
Year_fix = np.arange(2013, 2022)

# Perform cubic spline interpolation, filling the data from 2013 to 2021
model_max = interpolate.splrep(Year_real, CO2_max_real, k=3)
model_min = interpolate.splrep(Year_real, CO2_min_real, k=3)
model_mid = interpolate.splrep(Year_real, CO2_mid_real, k=3)
CO2_max_predict = interpolate.splev(np.arange(2013, 2022), model_max)
CO2_mid_predict = interpolate.splev(np.arange(2013, 2022), model_mid)
CO2_min_predict = interpolate.splev(np.arange(2013, 2022), model_min)

# Predictions made by ARIMA
model = ARIMA(CO2_max_predict, order=(1, 1, 1))
model_fit = model.fit()
# Predict the number from 2022 to 2050
forecastdata = model_fit.forecast(29)
CO2_max = np.hstack([CO2_max_predict, forecastdata])

model = ARIMA(CO2_min_predict, order=(1, 1, 1))
model_fit = model.fit()
# Predict the number from 2022 to 2050
forecastdata = model_fit.forecast(29)
CO2_min = np.hstack([CO2_min_predict, forecastdata])

model = ARIMA(CO2_mid_predict, order=(1, 1, 1))
model_fit = model.fit()
# Predict the number from 2022 to 2050
forecastdata = model_fit.forecast(29)
CO2_mid = np.hstack([CO2_mid_predict, forecastdata])

# Calculate the extra CO2
CO2_year = (CO2_max * S_rate_max + CO2_mid * S_rate_mid + CO2_min * S_rate_min - CO2_min) * Times * 365 * car_number / 100
CO2 = np.zeros(len(CO2_year))
CO2[0] = CO2_year[0]
for i in range(1, len(CO2_year)):
    CO2[i] = CO2[i - 1] + CO2_year[i]


# Calculation of other pollutant emissions
# CO
CO_max = np.zeros(len(year))
CO_mid = np.zeros(len(year))
CO_min = np.zeros(len(year))
CO_max[place] = CO_max_real
CO_mid[place] = CO_mid_real
CO_min[place] = CO_min_real

for i in range(2, 5):
    CO_max[i] = CO_max[1] - (CO_max[1] - CO_max[1] * Rate_CO[i] / Rate_CO[1]) / (CO_max[1] - CO_max[1] * Rate_CO[5] / Rate_CO[1]) * (CO_max[1] - CO_max[5])
    CO_mid[i] = CO_mid[1] - (CO_mid[1] - CO_mid[1] * Rate_CO[i] / Rate_CO[1]) / (CO_mid[1] - CO_mid[1] * Rate_CO[5] / Rate_CO[1]) * (CO_mid[1] - CO_mid[5])
    CO_min[i] = CO_min[1] - (CO_min[1] - CO_min[1] * Rate_CO[i] / Rate_CO[1]) / (CO_min[1] - CO_min[1] * Rate_CO[5] / Rate_CO[1]) * (CO_min[1] - CO_min[5])

for i in range(6, 8):
    CO_max[i] = CO_max[5] - (CO_max[5] - CO_max[5] * Rate_CO[i] / Rate_CO[5]) / (CO_max[5] - CO_max[5] * Rate_CO[8] / Rate_CO[5]) * (CO_max[5] - CO_max[8])
    CO_mid[i] = CO_mid[5] - (CO_mid[5] - CO_mid[5] * Rate_CO[i] / Rate_CO[5]) / (CO_mid[5] - CO_mid[5] * Rate_CO[8] / Rate_CO[5]) * (CO_mid[5] - CO_mid[8])
    CO_min[i] = CO_min[5] - (CO_min[5] - CO_min[5] * Rate_CO[i] / Rate_CO[5]) / (CO_min[5] - CO_min[5] * Rate_CO[8] / Rate_CO[5]) * (CO_min[5] - CO_min[8])

for i in range(9, len(year)):
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

for i in range(2, 5):
    HC_max[i] = HC_max[1] - (HC_max[1] - HC_max[1] * Rate_HC[i] / Rate_HC[1]) / (HC_max[1] - HC_max[1] * Rate_HC[5] / Rate_HC[1]) * (HC_max[1] - HC_max[5])
    HC_mid[i] = HC_mid[1] - (HC_mid[1] - HC_mid[1] * Rate_HC[i] / Rate_HC[1]) / (HC_mid[1] - HC_mid[1] * Rate_HC[5] / Rate_HC[1]) * (HC_mid[1] - HC_mid[5])
    HC_min[i] = HC_min[1] - (HC_min[1] - HC_min[1] * Rate_HC[i] / Rate_HC[1]) / (HC_min[1] - HC_min[1] * Rate_HC[5] / Rate_HC[1]) * (HC_min[1] - HC_min[5])

for i in range(6, 8):
    HC_max[i] = HC_max[5] - (HC_max[5] - HC_max[5] * Rate_HC[i] / Rate_HC[5]) / (HC_max[5] - HC_max[5] * Rate_HC[8] / Rate_HC[5]) * (HC_max[5] - HC_max[8])
    HC_mid[i] = HC_mid[5] - (HC_mid[5] - HC_mid[5] * Rate_HC[i] / Rate_HC[5]) / (HC_mid[5] - HC_mid[5] * Rate_HC[8] / Rate_HC[5]) * (HC_mid[5] - HC_mid[8])
    HC_min[i] = HC_min[5] - (HC_min[5] - HC_min[5] * Rate_HC[i] / Rate_HC[5]) / (HC_min[5] - HC_min[5] * Rate_HC[8] / Rate_HC[5]) * (HC_min[5] - HC_min[8])

for i in range(9, len(year)):
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

for i in range(2, 5):
    NOx_max[i] = NOx_max[1] - (NOx_max[1] - NOx_max[1] * Rate_NOx[i] / Rate_NOx[1]) / (NOx_max[1] - NOx_max[1] * Rate_NOx[5] / Rate_NOx[1]) * (NOx_max[1] - NOx_max[5])
    NOx_mid[i] = NOx_mid[1] - (NOx_mid[1] - NOx_mid[1] * Rate_NOx[i] / Rate_NOx[1]) / (NOx_mid[1] - NOx_mid[1] * Rate_NOx[5] / Rate_NOx[1]) * (NOx_mid[1] - NOx_mid[5])
    NOx_min[i] = NOx_min[1] - (NOx_min[1] - NOx_min[1] * Rate_NOx[i] / Rate_NOx[1]) / (NOx_min[1] - NOx_min[1] * Rate_NOx[5] / Rate_NOx[1]) * (NOx_min[1] - NOx_min[5])

for i in range(6, 8):
    NOx_max[i] = NOx_max[5] - (NOx_max[5] - NOx_max[5] * Rate_NOx[i] / Rate_NOx[5]) / (NOx_max[5] - NOx_max[5] * Rate_NOx[8] / Rate_NOx[5]) * (NOx_max[5] - NOx_max[8])
    NOx_mid[i] = NOx_mid[5] - (NOx_mid[5] - NOx_mid[5] * Rate_NOx[i] / Rate_NOx[5]) / (NOx_mid[5] - NOx_mid[5] * Rate_NOx[8] / Rate_NOx[5]) * (NOx_mid[5] - NOx_mid[8])
    NOx_min[i] = NOx_min[5] - (NOx_min[5] - NOx_min[5] * Rate_NOx[i] / Rate_NOx[5]) / (NOx_min[5] - NOx_min[5] * Rate_NOx[8] / Rate_NOx[5]) * (NOx_min[5] - NOx_min[8])

for i in range(9, len(year)):
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

for i in range(2, 5):
    PM_max[i] = PM_max[1] - (PM_max[1] - PM_max[1] * Rate_PM[i] / Rate_PM[1]) / (PM_max[1] - PM_max[1] * Rate_PM[5] / Rate_PM[1]) * (PM_max[1] - PM_max[5])
    PM_mid[i] = PM_mid[1] - (PM_mid[1] - PM_mid[1] * Rate_PM[i] / Rate_PM[1]) / (PM_mid[1] - PM_mid[1] * Rate_PM[5] / Rate_PM[1]) * (PM_mid[1] - PM_mid[5])
    PM_min[i] = PM_min[1] - (PM_min[1] - PM_min[1] * Rate_PM[i] / Rate_PM[1]) / (PM_min[1] - PM_min[1] * Rate_PM[5] / Rate_PM[1]) * (PM_min[1] - PM_min[5])

for i in range(6, 8):
    PM_max[i] = PM_max[5] - (PM_max[5] - PM_max[5] * Rate_PM[i] / Rate_PM[5]) / (PM_max[5] - PM_max[5] * Rate_PM[8] / Rate_PM[5]) * (PM_max[5] - PM_max[8])
    PM_mid[i] = PM_mid[5] - (PM_mid[5] - PM_mid[5] * Rate_PM[i] / Rate_PM[5]) / (PM_mid[5] - PM_mid[5] * Rate_PM[8] / Rate_PM[5]) * (PM_mid[5] - PM_mid[8])
    PM_min[i] = PM_min[5] - (PM_min[5] - PM_min[5] * Rate_PM[i] / Rate_PM[5]) / (PM_min[5] - PM_min[5] * Rate_PM[8] / Rate_PM[5]) * (PM_min[5] - PM_min[8])

for i in range(9, len(year)):
    PM_max[i] = PM_max[8] * (Rate_PM[8] - (Rate_PM[8] - Rate_PM[i])) / Rate_PM[8]
    PM_mid[i] = PM_mid[8] * (Rate_PM[8] - (Rate_PM[8] - Rate_PM[i])) / Rate_PM[8]
    PM_min[i] = PM_min[8] * (Rate_PM[8] - (Rate_PM[8] - Rate_PM[i])) / Rate_PM[8]



# 计算污染变化
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


# CO2 emission
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.18, right=0.90, left=0.2, hspace=0, wspace=0)
# Remove right and top borders
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# Set figure size
fig.set_size_inches(2500 / 900, 2500 / 900)
# Set axis label
plt.xlabel("Year", fontsize=14)
plt.ylabel("CO$_{2}$ emission (g)", fontsize=14)
# Set the font size and scale linewidth, with the scale inward
plt.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure
plt.plot(year, CO2_min, color=Color[0], label=Dis[0])
plt.plot(year, CO2_mid, color=Color[1], label=Dis[1])
plt.plot(year, CO2_max, color=Color[2], label=Dis[2])
# Set axis range
plt.ylim((0, 36))
plt.xlim((2013, 2050))
# Set axis scale
plt.yticks(np.array([9, 18, 27, 36]))
plt.xticks(np.array([2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=8, loc='upper right')
# Save figure
plt.savefig(r'D:\Fig5.png', dpi=900)
plt.close()


# CO emission
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.18, right=0.90, left=0.22, hspace=0, wspace=0)
# Remove right and top borders
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# Set figure size
fig.set_size_inches(2500 / 900, 2500 / 900)
# Set axis label
plt.xlabel("Year", fontsize=14)
plt.ylabel("CO emission (10$^{-1}$ g)", fontsize=14)
# Set the font size and scale linewidth, with the scale inward
plt.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure
plt.plot(year, CO_min * 10, color=Color[0], label=Dis[0])
plt.plot(year, CO_mid * 10, color=Color[1], label=Dis[1])
plt.plot(year, CO_max * 10, color=Color[2], label=Dis[2])
# Set axis range
plt.ylim((0, 2))
plt.xlim((2013, 2050))
# Set axis scale
plt.xticks(np.array([2020, 2030, 2040, 2050]))
plt.yticks(np.array([0.5, 1, 1.5, 2]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=8, loc='upper right')
# Save figure
plt.savefig(r'D:\Fig6.png', dpi=900)
plt.close()


# NOx emission
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.18, right=0.90, left=0.22, hspace=0, wspace=0)
# Remove right and top borders
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# Set figure size
fig.set_size_inches(2500 / 900, 2500 / 900)
# Set axis label
plt.xlabel("Year", fontsize=14)
plt.ylabel("NO$_{x}$ emission (10$^{-2}$ g)", fontsize=14)
# Set the font size and scale linewidth, with the scale inward
plt.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure
plt.plot(year, NOx_min * 100, color=Color[0], label=Dis[0])
plt.plot(year, NOx_mid * 100, color=Color[1], label=Dis[1])
plt.plot(year, NOx_max * 100, color=Color[2], label=Dis[2])
# Set axis range
plt.ylim((0, 2))
plt.xlim((2013, 2050))
# Set axis scale
plt.xticks(np.array([2020, 2030, 2040, 2050]))
plt.yticks(np.array([0.5, 1.0, 1.5, 2.0]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=8, loc='upper right')
# Save figure
plt.savefig(r'D:\Fig7.png', dpi=900)
plt.close()


# HC emission
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.18, right=0.90, left=0.22, hspace=0, wspace=0)
# Remove right and top borders
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# Set figure size
fig.set_size_inches(2500 / 900, 2500 / 900)
# Set axis label
plt.xlabel("Year", fontsize=14)
plt.ylabel("HC emission (10$^{-3}$ g)", fontsize=14)
# Set the font size and scale linewidth, with the scale inward
plt.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure
plt.plot(year, HC_min * 1000, color=Color[0], label=Dis[0])
plt.plot(year, HC_mid * 1000, color=Color[1], label=Dis[1])
plt.plot(year, HC_max * 1000, color=Color[2], label=Dis[2])
# Set axis range
plt.ylim((0, 6))
plt.xlim((2013, 2050))
# Set axis scale
plt.xticks(np.array([2020, 2030, 2040, 2050]))
plt.yticks(np.array([1.5, 3.0, 4.5, 6.0]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=8, loc='upper right')
# Save figure
plt.savefig(r'D:\Fig8.png', dpi=900)
plt.close()


# PM emission
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.18, right=0.90, left=0.22, hspace=0, wspace=0)
# Remove right and top borders
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# Set figure size
fig.set_size_inches(2500 / 900, 2500 / 900)
# Set axis label
plt.xlabel("Year", fontsize=14)
plt.ylabel("PM emission (10$^{-4}$ g)", fontsize=14)
# Set the font size and scale linewidth, with the scale inward
plt.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure
plt.plot(year, PM_min * 10000, color=Color[0], label=Dis[0])
plt.plot(year, PM_mid * 10000, color=Color[1], label=Dis[1])
plt.plot(year, PM_max * 10000, color=Color[2], label=Dis[2])
# Set axis range
plt.ylim((0, 6.4))
plt.xlim((2013, 2050))
# Set axis scale
plt.xticks(np.array([2020, 2030, 2040, 2050]))
plt.yticks(np.array([1.6, 3.2, 4.8, 6.4]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=8, loc='upper right')
# Save figure
plt.savefig(r'D:\Fig9.png', dpi=900)
plt.close()


# The change of SDAI proposition
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.18, right=0.95, left=0.15, hspace=0, wspace=0)
# Remove right and top borders
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# Set figure size
fig.set_size_inches(3800 / 900, 2500 / 900)
# Set axis label
plt.xlabel("Year", fontsize=14)
plt.ylabel("Driver type proportion (%)", fontsize=14)
# Set the font size and scale linewidth, with the scale inward
plt.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure
plt.plot(year, S_rate_min * 100, color=Color[0], label=Dis[0])
plt.plot(year, S_rate_mid * 100, color=Color[1], label=Dis[1])
plt.plot(year, S_rate_max * 100, color=Color[2], label=Dis[2])
# Set axis range
plt.ylim((0, 80))
plt.xlim((2013, 2015))
# Set axis scale
plt.yticks(np.array([20, 40, 60, 80]))
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=6, loc='upper right')
# Save figure
plt.savefig(r'D:\Fig10.png', dpi=900)
plt.close()


# The change of SDAI number
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.18, right=0.95, left=0.15, hspace=0, wspace=0)
# Remove right and top borders
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# Set figure size
fig.set_size_inches(3800 / 900, 2500 / 900)
# Set axis label
plt.xlabel("Year", fontsize=14)
plt.ylabel("Driver numbers (10$^{8}$)", fontsize=14)
# Set the font size and scale linewidth, with the scale inward
plt.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure
plt.plot(year, car_min / 10000, color=Color[0], label=Dis[0])
plt.plot(year, car_mid / 10000, color=Color[1], label=Dis[1])
plt.plot(year, car_max / 10000, color=Color[2], label=Dis[2])
plt.plot(year, Totalcar / 10000, color='k', linestyle='--', label='Total')
# Set axis range
plt.ylim((0, 8))
plt.xlim((2013, 2015))
# Set axis scale
plt.yticks(np.array([2, 4, 6, 8]))
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=6, loc='upper right')
# Save figure
plt.savefig(r'D:\Fig11.png', dpi=900)
plt.close()


# Total CO2 emission change
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# Remove top borders
axs.spines['top'].set_visible(False)
# Set figure size
fig.set_size_inches(4000 / 900, 3000 / 900)
# Set x-axis label
plt.xlabel("Year", fontsize=14)
# Set y-axis1 label
axs.set_ylabel("CO$_{2}$ emission (10$^{7}$ ton/year)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis1
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure of axis1
axs.bar(year, CO2_year / 10 ** 7 * Low / Center, color=Color[0], width=1, alpha=0.8, label='Lower bound', edgecolor='black', linewidth=0.3)
axs.bar(year, CO2_year / 10 ** 7 - CO2_year / 10 ** 7 * Low / Center, color=Color[1], width=1, alpha=0.8, bottom=CO2_year / 10 ** 7 * Low / Center, label='Predicted value', edgecolor='black', linewidth=0.3)
axs.bar(year, CO2_year / 10 ** 7 * Up / Center - CO2_year / 10 ** 7, color=Color[2], width=1, alpha=0.8, bottom=CO2_year / 10 ** 7, label='Upper bound', edgecolor='black', linewidth=0.3)
# Show Legend
plt.legend(fontsize=14, loc='upper left')
# Set axis1 range and scale
plt.xlim(2013,2050)
axs.set_ylim((0, 5))
axs.set_yticks(np.arange(1,6))
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
# Axis2
ax2 = axs.twinx()
ax2.spines['top'].set_visible(False)
ax2.set_ylabel("Accumulated CO$_{2}$ emission (10$^{8}$ ton)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis2
ax2.spines['bottom'].set_linewidth(0.3)
ax2.spines['right'].set_linewidth(0.3)
#  Plot figure of axis2
ax2.plot(year, CO2 / 10 ** 8, color='k', linewidth=1)
ax2.fill_between(year, CO2 / 10 ** 8 * Low / Center, CO2 / 10 ** 8 * Up / Center, color='grey', alpha=0.4, edgecolor='None')
# Set axis2 range and scale
ax2.set_ylim((0, 10))
ax2.set_yticks(np.arange(2, 11, 2))
# Save figure
plt.savefig(r'D:\Fig12.png', dpi=900)
plt.close()


# Annual change of total CO2 emission of single vehicle
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# Remove top borders
axs.spines['top'].set_visible(False)
# Set figure size
fig.set_size_inches(4000 / 900, 3000 / 900)
# Set x-axis label
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# Set y-axis1 label
axs.set_ylabel("Preventable CO$_{2}$ emission (%)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis1
axs.spines['bottom'].set_linewidth(1)
axs.spines['right'].set_linewidth(0.3)
# Plot figure of axis1
mid = (CO2_max * S_rate_max + CO2_mid * S_rate_mid + CO2_min * S_rate_min)
axs.bar(year, (mid - CO2_min) / (CO2_max - CO2_min) * 100, color='gray', width=1, alpha=0.2, label='Lower limit', edgecolor='black')
# Set axis1 range and scale
axs.set_ylim((20, 70))
axs.set_yticks(np.arange(30, 71, 10))
# Axis2
ax2 = axs.twinx()
ax2.spines['top'].set_visible(False)
ax2.set_ylabel("CO$_{2}$ emission (kg/veh·year)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis2
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['left'].set_linewidth(0.3)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
#  Plot figure of axis2
ax2.plot(year, CO2_min * Times * 365 / 10 ** 3, color=Color[0], linewidth=3, label=Dis[0])
ax2.plot(year, mid * Times * 365 / 10 ** 3, color=Color[1], linewidth=3, label='Predicted')
ax2.plot(year, CO2_max * Times * 365 / 10 ** 3, color=Color[2], linewidth=3, label=Dis[2])
# Set axis2 range and scale
ax2.set_ylim((0, 350))
ax2.set_yticks(np.arange(70, 351, 70))
# Show Legend
plt.legend(fontsize=14, loc='upper right')
# Save figure
plt.savefig(r'D:\Fig13.png', dpi=900)
plt.close()


# Calculation of median emission
CO_use_mid = (CO_max * S_rate_max + CO_mid * S_rate_mid + CO_min * S_rate_min) * Times * 365
HC_use_mid = (HC_max * S_rate_max + HC_mid * S_rate_mid + HC_min * S_rate_min) * Times * 365
NOx_use_mid = (NOx_max * S_rate_max + NOx_mid * S_rate_mid + NOx_min * S_rate_min) * Times * 365
PM_use_mid = (PM_max * S_rate_max + PM_mid * S_rate_mid + PM_min * S_rate_min) * Times * 365

# Total CO emission change
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# Remove top borders
axs.spines['top'].set_visible(False)
# Set figure size
fig.set_size_inches(4000 / 900, 3000 / 900)
# Set x-axis label
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# Set y-axis1 label
axs.set_ylabel("CO emission (10$^{5}$ ton/year)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis1
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure of axis1
axs.bar(year, CO_year / 10 ** 5 * Low / Center, color=Color[0], width=1, alpha=0.8, label='Lower bound', edgecolor='black', linewidth=0.3)
axs.bar(year, CO_year / 10 ** 5 - CO_year / 10 ** 5 * Low / Center, color=Color[1], width=1, alpha=0.8, bottom=CO_year / 10 ** 5 * Low / Center, label='Predicted value', edgecolor='black', linewidth=0.3)
axs.bar(year, CO_year / 10 ** 5 * Up / Center - CO_year / 10 ** 5, color=Color[2], width=1, alpha=0.8, bottom=CO_year / 10 ** 5, label='Upper bound', edgecolor='black', linewidth=0.3)
# Show Legend
plt.legend(fontsize=14, loc='upper left')
# Set axis1 range and scale
axs.set_ylim((0, 2))
axs.set_yticks(np.arange(0.4, 2.1, 0.4))
# Axis2
ax2 = axs.twinx()
ax2.spines['top'].set_visible(False)
ax2.set_ylabel("Accumulated CO emission (10$^{6}$ ton)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis2
ax2.spines['bottom'].set_linewidth(0.3)
ax2.spines['right'].set_linewidth(0.3)
# Plot figure of axis2
ax2.plot(year, CO_all / 10 ** 6, color='k', linewidth=1)
ax2.fill_between(year, CO_all / 10 ** 6 * Low / Center, CO_all / 10 ** 6 * Up / Center, color='grey', alpha = 0.4, edgecolor='None')
# Set axis2 range and scale
ax2.set_ylim((0, 3.5))
ax2.set_yticks(np.arange(0.7, 3.6, 0.7))
# Save figure
plt.savefig(r'D:\Fig14.png', dpi=900)
plt.close()


# Annual change of total CO emission of single vehicle
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# Remove top borders
axs.spines['top'].set_visible(False)
# Set figure size
fig.set_size_inches(4000 / 900, 3000 / 900)
# Set x-axis label
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# Set y-axis1 label
axs.set_ylabel("Preventable CO emission (%)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis1
axs.spines['bottom'].set_linewidth(1)
axs.spines['right'].set_linewidth(0.3)
# Plot figure of axis1
axs.bar(year, (CO_use_mid - CO_min * Times * 365) / (CO_max * Times * 365 - CO_min * Times * 365) * 100, color='gray', width=1, alpha=0.2, label='Lower limit', edgecolor='black')
# Set axis1 range and scale
axs.set_ylim((20, 70))
axs.set_yticks(np.arange(30, 71, 10))
# Axis2
ax2 = axs.twinx()
ax2.set_ylabel("CO emission (kg/veh·year)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis2
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['left'].set_linewidth(0.3)
ax2.spines['top'].set_visible(False)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure of axis2
ax2.plot(year, CO_min * Times * 365 / 1000, color=Color[0], linewidth=3, label=Dis[0])
ax2.plot(year, CO_use_mid / 1000, color=Color[1], linewidth=3, label='Predicted')
ax2.plot(year, CO_max * Times * 365 / 1000, color=Color[2], linewidth=3, label=Dis[2])
# Set axis2 range and scale
ax2.set_ylim((0, 1.5))
ax2.set_yticks(np.arange(0.3, 1.6, 0.3))
plt.legend(fontsize=14, loc='upper right')
# Save figure
plt.savefig(r'D:\Fig15.png', dpi=900)
plt.close()


# Total NOx emission change
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# Remove top borders
axs.spines['top'].set_visible(False)
# Set figure size
fig.set_size_inches(4000 / 900, 3000 / 900)
# Set x-axis label
plt.xlabel("Year", fontsize=14)
plt.xlim(2013, 2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# Set y-axis1 label
axs.set_ylabel("NO$_{x}$ emission (10$^{3}$ ton/year)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis1
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure of axis1
axs.bar(year, NOx_year / 10 ** 3 * Low / Center, color=Color[0], width=1, alpha=0.8, label='Lower bound', edgecolor='black', linewidth=0.3)
axs.bar(year, NOx_year / 10 ** 3 - NOx_year / 10 ** 3 * Low / Center, color=Color[1], width=1, alpha=0.8, bottom=NOx_year / 10 ** 3 * Low / Center, label='Predicted value', edgecolor='black', linewidth=0.3)
axs.bar(year, NOx_year / 10 ** 3 * Up / Center - NOx_year / 10 ** 3, color=Color[2], width=1, alpha=0.8, bottom=NOx_year / 10 ** 3, label='Upper bound', edgecolor='black', linewidth=0.3)
# Show Legend
plt.legend(fontsize=14, loc='upper left')
# Set axis1 range and scale
axs.set_ylim((0, 8))
axs.set_yticks(np.arange(1.6, 8.1, 1.6))
# Axis2
ax2 = axs.twinx()
ax2.spines['top'].set_visible(False)
ax2.set_ylabel("Accumulated NO$_{x}$ emission (10$^{5}$ ton)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis2
ax2.spines['bottom'].set_linewidth(0.3)
ax2.spines['right'].set_linewidth(0.3)
# Plot figure of axis2
ax2.plot(year, NOx_all / 10 ** 5, color='k', linewidth=1)
ax2.fill_between(year, NOx_all / 10 ** 5 * Low / Center, NOx_all / 10 ** 5 * Up / Center, color='grey', alpha = 0.4, edgecolor='None')
# Set axis2 range and scale
ax2.set_ylim((0, 1.5))
ax2.set_yticks(np.arange(0.3, 1.6, 0.3))
# Save figure
plt.savefig(r'D:\Fig16.png', dpi=900)
plt.close()


# Annual change of total NOx emission of single vehicle
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# Remove top borders
axs.spines['top'].set_visible(False)
# Set figure size
fig.set_size_inches(4000 / 900, 3000 / 900)
# Set x-axis label
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# Set y-axis1 label
axs.set_ylabel("Preventable NO$_{x}$ emission (%)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis1
axs.spines['bottom'].set_linewidth(1)
axs.spines['right'].set_linewidth(0.3)
# Plot figure of axis1
axs.bar(year, (NOx_use_mid - NOx_min * Times * 365) / (NOx_max * Times * 365 - NOx_min * Times * 365) * 100, color='gray', width=1, alpha=0.2, label='Lower limit', edgecolor='black')
# Set axis1 range and scale
axs.set_ylim((20, 70))
axs.set_yticks(np.arange(30, 71, 10))
# Axis2
ax2 = axs.twinx()
ax2.set_ylabel("NO$_{x}$ emission (10$^{-2}$kg/veh·year)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis2
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['left'].set_linewidth(0.3)
ax2.spines['top'].set_visible(False)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure of axis2
ax2.plot(year, NOx_min * Times * 365 / 10, color=Color[0], linewidth=3, label=Dis[0])
ax2.plot(year, NOx_use_mid / 10, color=Color[1], linewidth=3, label='Predicted')
ax2.plot(year, NOx_max * Times * 365 / 10, color=Color[2], linewidth=3, label=Dis[2])
# Plot figure of axis2
ax2.set_ylim((0, 10))
ax2.set_yticks(np.arange(2, 10.1, 2))
# Show Legend
plt.legend(fontsize=14, loc='upper right')
# Save figure
plt.savefig(r'D:\Fig17.png', dpi=900)
plt.close()


# Total HC emission change
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# Remove top borders
axs.spines['top'].set_visible(False)
# Set figure size
fig.set_size_inches(4000 / 900, 3000 / 900)
# Set x-axis label
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# Set y-axis1 label
axs.set_ylabel("HC emission (10$^{3}$ ton/year)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis1
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure of axis1
axs.bar(year, HC_year / 10 ** 3 * Low / Center, color=Color[0], width=1, alpha=0.8, label='Lower bound', edgecolor='black', linewidth=0.3)
axs.bar(year, HC_year / 10 ** 3 - HC_year / 10 ** 3 * Low / Center, color=Color[1], width=1, alpha=0.8, bottom=HC_year / 10 ** 3 * Low / Center, label='Predicted value', edgecolor='black', linewidth=0.3)
axs.bar(year, HC_year / 10 ** 3 * Up / Center - HC_year / 10 ** 3, color=Color[2], width=1, alpha=0.8, bottom=HC_year / 10 ** 3, label='Upper bound', edgecolor='black', linewidth=0.3)
# Show Legend
plt.legend(fontsize=14, loc='upper left')
# Set axis1 range and scale
axs.set_ylim((0, 3))
axs.set_yticks(np.arange(0.6, 3.1, 0.6))
# Axis2
ax2 = axs.twinx()
ax2.spines['top'].set_visible(False)
ax2.set_ylabel("Accumulated HC emission (10$^{4}$ ton)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis2
ax2.spines['bottom'].set_linewidth(0.3)
ax2.spines['right'].set_linewidth(0.3)
# Plot figure of axis2
ax2.plot(year, HC_all / 10 ** 4, color='k', linewidth=1)
ax2.fill_between(year, HC_all / 10 ** 4 * Low / Center, HC_all / 10 ** 4 * Up / Center, color='grey', alpha = 0.4, edgecolor='None')
# Set axis2 range and scale
ax2.set_ylim((0, 3.5))
ax2.set_yticks(np.arange(0.9,4.6,0.9))
# Save figure
plt.savefig(r'D:\Fig18.png', dpi=900)
plt.close()


# Annual change of total HC emission of single vehicle
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# Remove top borders
axs.spines['top'].set_visible(False)
# Set figure size
fig.set_size_inches(4000 / 900, 3000 / 900)
# Set x-axis label
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# Set y-axis1 label
axs.set_ylabel("Preventable HC emission (%)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis1
axs.spines['bottom'].set_linewidth(1)
axs.spines['right'].set_linewidth(0.3)
# Plot figure of axis1
axs.bar(year, (HC_use_mid - HC_min * Times * 365) / (HC_max * Times * 365 - HC_min * Times * 365) * 100, color='gray', width=1, alpha=0.2, label='Lower limit', edgecolor='black')
# Set axis1 range and scale
axs.set_ylim((20, 70))
axs.set_yticks(np.arange(30, 71, 10))
# Axis2
ax2 = axs.twinx()
ax2.set_ylabel("HC emission (10$^{-2}$kg/veh·year)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis2
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['left'].set_linewidth(0.3)
ax2.spines['top'].set_visible(False)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure of axis2
ax2.plot(year, HC_min * Times * 365 / 10, color=Color[0], linewidth=3, label=Dis[0])
ax2.plot(year, HC_use_mid / 10, color=Color[1], linewidth=3, label='Predicted')
ax2.plot(year, HC_max * Times * 365 / 10, color=Color[2], linewidth=3, label=Dis[2])
# Set axis2 range and scale
ax2.set_ylim((0, 3))
ax2.set_yticks(np.arange(0.6, 3.1, 0.6))
# Show Legend
plt.legend(fontsize=14, loc='upper right')
# Save figure
plt.savefig(r'D:\Fig19.png', dpi=900)
plt.close()


# Total PM emission change
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# Remove top borders
axs.spines['top'].set_visible(False)
# Set figure size
fig.set_size_inches(4000 / 900, 3000 / 900)
# Set x-axis label
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# Set y-axis1 label
axs.set_ylabel("PM emission (10$^{2}$ ton/year)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis1
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure of axis1
axs.bar(year, PM_year / 10 ** 2 * Low / Center, color=Color[0], width=1, alpha=0.8, label='Lower bound', edgecolor='black', linewidth=0.3)
axs.bar(year, PM_year / 10 ** 2 - PM_year / 10 ** 2 * Low / Center, color=Color[1], width=1, alpha=0.8, bottom=PM_year / 10 ** 2 * Low / Center, label='Predicted value', edgecolor='black', linewidth=0.3)
axs.bar(year, PM_year / 10 ** 2 * Up / Center - PM_year / 10 ** 2, color=Color[2], width=1, alpha=0.8, bottom=PM_year / 10 ** 2, label='Upper bound', edgecolor='black', linewidth=0.3)
# Show Legend
plt.legend(fontsize=14, loc='upper left')
# Set axis1 range and scale
axs.set_ylim((0, 5))
axs.set_yticks(np.arange(1,5.1,1))
# Axis2
ax2 = axs.twinx()
ax2.spines['top'].set_visible(False)
ax2.set_ylabel("Accumulated PM emission (10$^{3}$ ton)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis2
ax2.spines['bottom'].set_linewidth(0.3)
ax2.spines['right'].set_linewidth(0.3)
# Plot figure of axis2
ax2.plot(year, PM_all / 10 ** 3, color='k', linewidth=1)
ax2.fill_between(year, PM_all / 10 ** 3 * Low / Center, PM_all / 10 ** 3 * Up / Center, color='grey', alpha=0.4, edgecolor='None')
# Set axis2 range and scale
ax2.set_ylim((0, 7))
ax2.set_yticks(np.arange(1.4, 7.1, 1.4))
# Save figure
plt.savefig(r'D:\Fig20.png', dpi=900)
plt.close()


# Annual change of total PM emission of single vehicle
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.90, bottom=0.13, right=0.85, left=0.15, hspace=0, wspace=0)
# Remove top borders
axs.spines['top'].set_visible(False)
# Set figure size
fig.set_size_inches(4000 / 900, 3000 / 900)
# Set x-axis label
plt.xlabel("Year", fontsize=14)
plt.xlim(2013,2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# Set y-axis1 label
axs.set_ylabel("Preventable PM emission (%)", fontsize=14)
axs.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis1
axs.spines['bottom'].set_linewidth(1)
axs.spines['right'].set_linewidth(0.3)
# Plot figure of axis1
axs.bar(year, (PM_use_mid - PM_min * Times * 365) / (PM_max * Times * 365 - PM_min * Times * 365) * 100, color='gray', width=1, alpha=0.2, label='Lower limit', edgecolor='black')
# Set axis1 range and scale
axs.set_ylim((20, 70))
axs.set_yticks(np.arange(30, 71, 10))
# Axis2
ax2 = axs.twinx()
ax2.set_ylabel("PM emission (10$^{-3}$kg/veh·year)", fontsize=14)
ax2.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis2
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['left'].set_linewidth(0.3)
ax2.spines['top'].set_visible(False)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# 输出图像
ax2.plot(year, PM_min * Times * 365, color=Color[0], linewidth=3, label=Dis[0])
ax2.plot(year, PM_use_mid, color=Color[1], linewidth=3, label='Predicted')
ax2.plot(year, PM_max * Times * 365, color=Color[2], linewidth=3, label=Dis[2])
# Set axis2 range and scale
ax2.set_ylim((0, 3))
ax2.set_yticks(np.arange(0.6, 3.1, 0.6))
# Show Legend
plt.legend(fontsize=14, loc='upper right')
# Save figure
plt.savefig(r'D:\Fig21.png', dpi=900)
plt.close()
