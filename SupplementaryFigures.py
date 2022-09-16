import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set Color
Color = ['#7079DE', '#FC6F68', '#FFB64D']

# Read data
data0 = pd.read_csv('CongestionTimes.csv')
Number = np.asarray(data0.iloc[:, 0])
Times = np.asarray(data0.iloc[:, 1])

s = []
for i in range(0,len(Number)):
    s = np.append(s, Number[i] * np.ones(Times[i]))

# Figure Supplementary10
fig, axs = plt.subplots(1, 1, dpi=900)
plt.subplots_adjust(top=0.95, bottom=0.1, right=0.95, left=0.07, hspace=0, wspace=0)
# Remove right and top borders
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# Set figure size
fig.set_size_inches(10000 / 900, 6000 / 900)
# Set tick mark to inward
plt.tick_params(labelsize=15,direction='in')
plt.hist(s, 30, color=Color[0], edgecolor='black', linewidth=0.3, width=2)
# plt.plot((3,3), (0, gvf[2]), linewidth=1.0, color='k', linestyle='--')
# plt.plot((0,3), (gvf[2], gvf[2]), linewidth=1.0, color='k', linestyle='--')
# Set axis
plt.xlabel("Response number", fontsize=15)
plt.ylabel("Frequency", fontsize=15)
# Set axis range
plt.xlim((40, 120))
plt.ylim((0, 50))
# Only the coordinate axis in the y direction is turned on
plt.grid(axis='y')
# Set axis scale
plt.xticks(np.arange(40,121,20))
plt.yticks(np.arange(0,51,10))
# Save figure
plt.savefig(r'D:\Supplementary10.png', dpi=900)
plt.close()
