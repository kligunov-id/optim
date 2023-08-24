import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

sns.set()

data_s = np.array([9.16, 9.13, 9.18, 9.17, 9.21, 9.27, 9.30, 9.35])
data = 100 - data_s / 9.75 * 100
scales = 2 ** np.arange(8)
errors = np.array([0.25, 0.24, 0.22, 0.22, 0.21, 0.19, 0.19, 0.17]) / 1000 ** 0.5

yticks = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5])
plt.xscale("log", base=2)
plt.xticks(scales, [f"{scale}\n{scale * 1e5:.1e}" for scale in scales], fontsize=14)
plt.yticks(yticks, [f"{ytick}" for ytick in yticks], fontsize=16)
plt.plot(scales, data)
#plt.fill_between(scales, 100 - (data_s + errors) / 9.75 * 100, 100 - (data_s - errors) / 9.75 * 100, alpha=0.2)
plt.xlabel("Scale / Num. Parameters", fontsize=18)
plt.ylabel("Perfomance gap", fontsize=18)
#plt.axhline(y=9.75, color="green", linestyle='dashed', label="Optimal")
plt.axhline(y=100 - 9.11 / 9.75 * 100, color="red", linestyle='dashed', label="Greedy")
plt.legend()
