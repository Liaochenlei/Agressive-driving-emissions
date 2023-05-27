import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Data path
path = r'data\CongestionTimes.csv'
# Ouput path
path_out = r'Figure'


# Read data
data0 = pd.read_csv(path)
Number = np.asarray(data0.iloc[:, 0])
Times = np.asarray(data0.iloc[:, 1])

s = []
for i in range(0, len(Number)):
    s = np.append(s, Number[i] * np.ones(Times[i]))


# Create the folder if not exist
if not os.path.exists(path_out):
    os.mkdir(path_out)

# Figure Supplementary10
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.1, right=0.95, left=0.07, hspace=0, wspace=0)
# Remove right and top borders
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# Set figure size
fig.set_size_inches(10000 / 900, 6000 / 900)
# Set tick mark to inward
plt.tick_params(labelsize=15, direction='in')
plt.hist(s, 30, color='#7079DE', edgecolor='black', linewidth=0.3, width=2)
# Set axis
plt.xlabel("Response number", fontsize=15)
plt.ylabel("Frequency", fontsize=15)
# Set axis range
plt.xlim((40, 120))
plt.ylim((0, 50))
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y')
# Set axis scale
plt.xticks(np.arange(40, 121, 20))
plt.yticks(np.arange(0, 51, 10))
# Save figure
plt.savefig(path_out + '\\' + 'CongestionTimes.png', dpi=900)
plt.close()
