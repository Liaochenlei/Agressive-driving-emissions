import pandas as pd
from math import e
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import interpolate
import matplotlib.pyplot as plt


################################################
# The following is the filling missing data and forecasting increment
# Population in 10k
Year_old = np.arange(2012, 2021)
Population = np.array([135404, 136072, 136782, 137462, 138271, 139008, 139538, 140005, 141178])
# Vehicle increment, 2013-2020
Car = np.array([2017, 2188, 2385, 2752, 2813, 2673, 2578, 2424])

# Population prediction made by UN (Unit: k)
# Due to the lack of vehicle data in Taiwan, Hong Kong and Macao, the corresponding population should also be removed
Year_use1 = np.arange(2020, 2061, 5)
# Population from 2020 to 2060
Population_use = np.array([1407360.669, 1425493.663, 1431578.343, 1428292.369, 1416507.141, 1397321.995, 1371112.769, 1339066.038, 1303275.936])

# GDP made by PWC in PPP (Unit: trillion)
Year_use2 = np.array([2016, 2020, 2025, 2030, 2035, 2040, 2045, 2050])
GDP_use_PPP = np.array([21.3, 26.9, 32.7, 38.0, 42.7, 47.4, 52.9, 58.5])
# Perform cubic spline interpolation
Year1 = np.arange(2020, 2051)
Year2 = np.arange(2015, 2051)
tck_Population = interpolate.splrep(Year_use1, Population_use, k=3)
tck_GDP = interpolate.splrep(Year_use2, GDP_use_PPP, k=3)
Population_new = interpolate.splev(Year1, tck_Population)
GDP_out = interpolate.splev(Year2, tck_GDP)

"""
# Plt figure
plt.plot(Year_use1, Population_use, 'o', Year1, Population_new)
plt.show()
plt.plot(Year_use2, GDP_use_PPP, 'o', Year2, GDP_out)
plt.show()
"""

# Combine the predicted population from 2015 to 2020 (Unit:k)
Population_out = np.hstack((Population[3:8] * 10, Population_new[:]))

# Calculating GDP and population growth rate, 2021-2050
GDP_speed = np.zeros(35)
Population_speed = np.zeros(35)
for i in range(0, 35):
    GDP_speed[i] = (GDP_out[i + 1] - GDP_out[i]) / GDP_out[i]
    Population_speed[i] = (Population_out[i + 1] - Population_out[i]) / Population_out[i]

# New vehicle increment, 2016-2020 (Unit: 10k)
Car_increase_real = np.array([2752, 2813, 2673, 2578, 2424])

# Fitting increment
X = np.array(np.vstack([GDP_speed[0:5], Population_speed[0:5]])).reshape(5, 2)
Y = Car_increase_real.reshape(5, 1)

# Binary linear regression
model = LinearRegression()
model.fit(X, Y)
a = model.coef_  # Regression coefficient
b = model.intercept_  # Intercept
score = model.score(X, Y)  # R-test
# print(a,b,score)

# Forecast vehicle increment
Car_increase_predict = a[0, 0] * GDP_speed + a[0, 1] * Population_speed + b
Car_increase_predict[0:6] = np.array([2752, 2813, 2673, 2578, 2424, 2622])
# print(Car_increase_predict)
########################################################


########################################################
# The following is iteration
# Total number of vehicles 2013-2050 (Unit:10k)
Car_number = np.zeros(38)
Car_number[0:9] = np.array([13693, 15400, 17181, 19393, 21697, 23982, 26150, 28120, 30247])
# Total number and proportion of electric vehicles 2013-2050 (Unit:10k)
# The proportion of electric vehicles will become 100% after 2035
EV_number = np.zeros(38)
EV_number[0:9] = np.array([0, 22, 42, 91, 153, 266.74, 381, 492, 784])
# Using cubic spline interpolation to prediction vehicle proportion
EV_real = np.array([0, 22, 26, 49, 65, 113, 120, 187, 295]) / np.array([2017, 2188, 2385, 2752, 2813, 2673, 2578, 2424, 2622])
model = interpolate.splrep(np.hstack([np.arange(2019, 2022), np.array(2035)]), np.hstack([EV_real[6:], np.array(1)]), k=3)
EV_predict = interpolate.splev(np.arange(2019, 2036), model)
EV_ratio = np.hstack([EV_real[0:6], EV_predict, np.ones(15)])

"""
# Plt figure
plt.plot(np.arange(2013, 2051), EV_ratio)
plt.show()
"""

# Number of vehicles of each generation 2013-2050
Car_general = np.zeros([38, 8])
Car_general[0] = np.array([0.033, 0.101, 0.115, 0.360, 0.391, 0, 0, 0]) * (Car_number[0] - EV_number[0])
Car_general[1] = np.array([0.021, 0.082, 0.093, 0.318, 0.452, 0.034, 0, 0]) * (Car_number[1] - EV_number[1])
Car_general[2] = np.array([0.014, 0.068, 0.077, 0.278, 0.501, 0.062, 0, 0]) * (Car_number[2] - EV_number[2])
Car_general[3] = np.array([0.010, 0.054, 0.064, 0.243, 0.524, 0.105, 0, 0]) * (Car_number[3] - EV_number[3])
Car_general[4] = np.array([0.001, 0.037, 0.055, 0.217, 0.475, 0.215, 0, 0]) * (Car_number[4] - EV_number[4])
Car_general[5] = np.array([0.000, 0.030, 0.045, 0.191, 0.425, 0.309, 0, 0]) * (Car_number[5] - EV_number[5])
# Gen.0 - Gen.6b and EV, 2013-2050, Vehicle age 0-29
Car_ratio = np.zeros([9, 38, 30])
# Vehicle age 0-29
# Read data
data = pd.read_csv('VehicleAgeGenerationDistribution.csv')
for i in range(0, 9):
    for j in range(0, 9):
        Car_ratio[i, j, :] = ID = np.hstack([np.asarray(data.iloc[:, i * 9 + j]), np.zeros(6)])

# Fit the survival rate of the reference paper
Year_fix = np.array([5, 7, 10, 12, 15, 17]).reshape(6, 1)
Car_ratio_2012 = np.array([508, 510, 289, 155, 31, 11]).reshape(6, 1) / 528
Car_ratio_2014 = np.array([524, 512, 416, 277, 70, 21]).reshape(6, 1) / 528
Car_ratio_2016 = np.array([524, 512, 437, 325, 114, 44]).reshape(6, 1) / 528
b = np.zeros(39)
c = np.zeros(39)
# Simple linear regression
# 2012
model = LinearRegression()
model.fit(Year_fix, np.log(1 / Car_ratio_2012 - 1))
b[0] = np.exp(model.intercept_)  # Intercept
c[0] = model.coef_[0]  # Regression coefficient
# score = model.score(Year_fix, np.log(1 / Car_ratio_2012 - 1))  # R-test
# 2014
model = LinearRegression()
model.fit(Year_fix, np.log(1 / Car_ratio_2014 - 1))
b[2] = np.exp(model.intercept_)
c[2] = model.coef_[0]
# score = model.score(Year_fix, np.log(1 / Car_ratio_2014 - 1))
# 2016
model = LinearRegression()
model.fit(Year_fix, np.log(1 / Car_ratio_2016 - 1))
b[4] = np.exp(model.intercept_)
c[4] = model.coef_[0]
# score = model.score(Year_fix, np.log(1 / Car_ratio_2016 - 1))

# Take the median value for 2013 and 2015
Car_ratio_2013 = (Car_ratio_2012 + Car_ratio_2014) / 2
Car_ratio_2015 = (Car_ratio_2014 + Car_ratio_2016) / 2
# 2013
model = LinearRegression()
model.fit(Year_fix, np.log(1 / Car_ratio_2013 - 1))
b[1] = np.exp(model.intercept_)
c[1] = model.coef_[0]
# score = model.score(Year_fix, np.log(1 / Car_ratio_2013 - 1))
# 2015
model = LinearRegression()
model.fit(Year_fix, np.log(1 / Car_ratio_2015 - 1))
b[3] = np.exp(model.intercept_)
c[3] = model.coef_[0]
# score = model.score(Year_fix, np.log(1 / Car_ratio_2015 - 1))


# Translate at the speed from Distance * (1 - 1/(1+e** -(0.05 * i)) after 2016, 2017-2050
Year = np.arange(0, 30)

Solve = np.array([0.9, 0.7, 0.5, 0.3, 0.1]).reshape(5, 1)
Year_fix = np.array([9.201635, 11.445915, 12.854565, 14.263215, 16.507495]).reshape(5, 1)
Distance = np.array([1.49788, 1.41214, 1.35831, 1.3045, 1.21875]).reshape(5, 1) / 2

for i in range(1, 35):
    Year_fix = Year_fix + Distance * (1 - 1 / (1 + e ** -(0.05 * i)))
    model = LinearRegression()
    model.fit(Year_fix, np.log(1 / Solve - 1))
    b[i + 4] = np.exp(model.intercept_)  # Intercept
    c[i + 4] = model.coef_[0]  # Regression coefficient
    # score = model.score(Year_fix, np.log(1 / Car_ratio_2016 - 1))  # R-test
    # plt.plot(Year_fix, Car_ratio_2016, 'o', Year, 1 / (1 + b[i + 4] * e ** (c[i + 4] * Year)))

# Calculate the survival rate of 2013-2050
Car_survival = np.zeros([38, 29])
for i in range(1, 39):
    Car_survival_now = 1 / (1 + b[i] * e ** (c[i] * Year))
    Car_survival[i - 1] = np.hstack([Car_survival_now[0], np.divide(Car_survival_now[1:-1], Car_survival_now[0:-2])])

"""
# Plt figure
for i in range(0, 38):
    plt.plot(np.arange(0, 31), 1 / (1 + b[i] * e ** (c[i] * np.arange(0, 31))))
plt.ylim(0,1)
plt.show()
"""

# Starting iteration, 2022-2050
for i in range(9, 38):
    # First, scraping the old vehicle
    for j in range(0, 9):
        Car_ratio[j, i, :] = np.hstack([np.array(0), Car_ratio[j, i - 1, :-1] * Car_survival[i - 1]])
    # Increase new vehicle
    # Increase the electric new vehicle
    Car_ratio[8, i, 0] = Car_increase_predict[i - 3] * EV_ratio[i]
    # In 2022, Gen.6a account for 1/3, Gen.6b account for 2/3, and then all Gen.6b
    if i == 2022:
        Car_ratio[7, i, 0] = Car_increase_predict[i - 3] * (1 - EV_ratio[i]) * 2 / 3
        Car_ratio[6, i, 0] = Car_increase_predict[i - 3] * (1 - EV_ratio[i]) / 3
    else:
        Car_ratio[7, i, 0] = Car_increase_predict[i - 3] * (1 - EV_ratio[i])

# Calculate the total number of each generation
Sum = np.zeros([9, 38])
for i in range(0, 38):
    for j in range(0, 9):
        Sum[j, i] = np.sum(Car_ratio[j, i])

# Calculate total number
for i in range(0, 38):
    Car_number[i] = np.sum(Sum[:, i])

# Calculate average vehicle age
Car_age = np.zeros(38)
for i in range(0, 38):
    for j in range(0, 9):
        for k in range(0, 30):
            Car_age[i] = Car_age[i] + k * Car_ratio[j, i, k] / Car_number[i]

# Output the number by vehicle age
Car_age_year = np.zeros([38, 30])
for i in range(0, 38):
    for j in range(0, 30):
        Car_age_year[i, j] = Car_age_year[i, j] + np.sum(Car_ratio[:, i, j])

# print(Car_age)
# print(Car_number)
# print(Car_number / np.hstack((Population[1:8] * 10, Population_new[:])) * 10)
# print(np.hstack((Population[1:8] * 10, Population_new[:])))

##############################################################


##############################################################
# Set color
Color = ['#000056', '#3537DE', '#7079DE', '#9168CE', '#D15B7E', '#FC6F68', '#FFB36A', '#FFDA43', '#63E5B3']
plt.rcParams['font.sans-serif'] = ['Arial']
# Calculated proportion
Ratio = np.zeros([9, 38])
for i in range(0, 38):
    Ratio[:, i] = Sum[:, i] / Car_number[i]

# Plt figure
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.97, bottom=0.13, right=0.90, left=0.13, hspace=0, wspace=0)
# Remove top borders
# axs.spines['top'].set_visible(False)
# Set figure size
fig.set_size_inches(4500 / 900, 3900 / 900)
# Set x-axis label
plt.xlabel("Year", fontsize=14)
plt.xlim(2013, 2050)
plt.xticks(np.array([2013, 2020, 2030, 2040, 2050]))
# Set y-axis1 label
axs.set_ylabel("Ratio of different emission standard (%)", fontsize=14)
axs.tick_params(labelsize=14, width=1, direction='in')
# Change the linewidth of axis1
# axs.spines['bottom'].set_linewidth(1)
# axs.spines['right'].set_linewidth(0.3)
# Plot figure of axis1
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
# Set axis1 range and scale
axs.set_ylim((0, 100))
axs.set_yticks(np.arange(20, 100.1, 20))
# Axis2
ax2 = axs.twinx()
ax2.set_ylabel("Average vehicle age (year)", fontsize=14)
ax2.tick_params(labelsize=14, width=1, direction='in')
# Change the linewidth of axis2
# ax2.spines['bottom'].set_linewidth(1)
# ax2.spines['left'].set_linewidth(0.3)
# ax2.spines['top'].set_visible(False)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure of axis2
ax2.plot(np.arange(2013, 2051), Car_age, color='k', linewidth=2)
# Set axis2 range and scale
ax2.set_ylim((0, 15))
ax2.set_yticks(np.arange(3, 15.1, 3))
# Show Legend
# plt.legend(fontsize=12, loc='upper right')
# Save figure
plt.savefig(r'D:\Fig0.png', dpi=900)
plt.close()


# Car increment
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.19, right=0.91, left=0.21, hspace=0, wspace=0)
# Remove top borders
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# Set figure size
fig.set_size_inches(2800 / 900, 2100 / 900)
# Set axis label
plt.xlabel("Year", fontsize=14)
plt.ylabel("Vehicle sales (10$^{7}$)", fontsize=14)
# Set the font size and scale linewidth, with the scale inward
plt.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure
plt.bar(np.arange(2013, 2051), np.hstack([np.array([2017, 2188, 2385]), Car_increase_predict]) / 1000, color='#FFB64D')
# Set axis range
plt.ylim((1.5, 3))
plt.xlim((2013, 2050))
# Set axis scale
plt.yticks(np.array([1.8, 2.1, 2.4, 2.7, 3]))
plt.xticks(np.array([2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=8, loc='upper right')
# Save figure
plt.savefig(r'D:\Fig1.png', dpi=900)
plt.close()


# EV increment
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.19, right=0.92, left=0.25, hspace=0, wspace=0)
# Remove right and top borders
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# Set figure size
fig.set_size_inches(2500 / 900, 2100 / 900)
# Set axis label
plt.xlabel("Year", fontsize=14)
plt.ylabel("EV of vehicle sales (%)", fontsize=14)
# Set the font size and scale linewidth, with the scale inward
plt.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure
plt.plot(np.arange(2013, 2051), EV_ratio * 100, color='#7079DE', linewidth=2)
plt.plot((2035, 2035), (0, 100), color='grey', linestyle='--', linewidth=2)
# Set axis range
plt.ylim((0, 100))
plt.xlim((2013, 2050))
# Set axis scale
plt.yticks(np.array([20, 40, 60, 80, 100]))
plt.xticks(np.array([2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=8, loc='upper right')
# Save figure
plt.savefig(r'D:\Fig2.png', dpi=900)
plt.close()


# Set color
Color = ['k', '#000089', '#3537DE', '#7079DE', '#9168CE', '#D15B7E', '#FC6F68', '#FFA06A', '#FFC36A', '#FFDA43', '#FFF89E']
Color16 = ['#7079DE', '#D15B7E', '#FC6F68', '#FFA06A', '#FFB64D', '#FFDA43', '#FFE88E']
Color10 = np.array([[112, 121, 222], [209, 91, 126], [252, 111, 104], [255, 160, 106], [255, 182, 77], [255, 218, 67], [255, 232, 142]])
# Survival rate
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.19, right=0.92, left=0.25, hspace=0, wspace=0)
# Remove right and top borders
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# Set figure size
fig.set_size_inches(2500 / 900, 2100 / 900)
# Set axis label
plt.xlabel("Year", fontsize=14)
plt.ylabel("Survival rate (%)", fontsize=14)
# Set the font size and scale linewidth, with the scale inward
plt.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure
for i in range(0, 38):
    # Calculate the current color
    flag = min(int(i / 6), 5)   # Used to indicate color position
    if i >= 32:  # Determine interval length
        Len = 7
    else:
        Len = 6
    # Determine RGB
    Red = int(Color10[flag, 0] + (Color10[flag + 1, 0] - Color10[flag, 0]) * (i - 6 * flag) / Len)
    Green = int(Color10[flag, 1] + (Color10[flag + 1, 1] - Color10[flag, 1]) * (i - 6 * flag) / Len)
    Blue = int(Color10[flag, 2] + (Color10[flag + 1, 2] - Color10[flag, 2]) * (i - 6 * flag) / Len)
    # Convert to Color Code
    Color_now = '#' + str(hex(Red)[2:]) + str(hex(Green)[2:]) + str(hex(Blue)[2:])
    plt.plot(np.arange(0, 31, 0.1), 100 / (1 + b[i] * e ** (c[i] * np.arange(0, 31, 0.1))), color=Color_now, linewidth=0.4)
# Set axis range
plt.ylim((0, 100))
plt.xlim((0, 31))
# Set axis scale
plt.yticks(np.array([20, 40, 60, 80, 100]))
plt.xticks(np.array([0, 5, 10, 15, 20, 25, 30]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=8, loc='upper right')
# Save figure
plt.savefig(r'D:\Fig3.png', dpi=900)
plt.close()


# Car ownership
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.19, right=0.92, left=0.25, hspace=0, wspace=0)
# Remove right and top borders
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# Set figure size
fig.set_size_inches(2500 / 900, 2100 / 900)
# Set axis label
plt.xlabel("Year", fontsize=14)
plt.ylabel("Vehicle ownership (10$^{8}$)", fontsize=14)
# Set the font size and scale linewidth, with the scale inward
plt.tick_params(labelsize=14, width=0.3, direction='in')
# Change the linewidth of axis
axs.spines['bottom'].set_linewidth(0.3)
axs.spines['left'].set_linewidth(0.3)
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y', linewidth=0.2)
# Plot figure
plt.plot(np.arange(2013, 2051), Car_number / 10000, color='#FC6F68', linewidth=2)
# Set axis range
plt.ylim((0, 6))
plt.xlim((2013, 2050))
# Set axis scale
plt.yticks(np.array([1.2, 2.4, 3.6, 4.8, 6]))
plt.xticks(np.array([2020, 2030, 2040, 2050]))
# plt.xticks(rotation=20)
# plt.legend(fontsize=8, loc='upper right')
# Save figure
plt.savefig(r'D:\Fig4.png', dpi=900)
plt.close()
