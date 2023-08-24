import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set()
scores = pd.read_csv("scores.csv")
scores = (1 - scores / scores.loc["Optimal"]) * 100 # perfomance gap in percents

greedy_treshold = scores.loc["Greedy"]
scores.drop(index=["Optimal", "Random"], inplace=True)

positions = pd.DataFrame(index=scores.index)
for dataset in scores.columns:
    positions[dataset] = np.argsort(np.argsort(scores[dataset]))

bar_width = 1 / (len(scores.index) + 1)
base_positions = np.arange(len(scores.columns))

for model in scores.index:
    bars_positions = positions.loc[model] * bar_width + base_positions
    plt.bar(x=bars_positions, height=scores.loc[model], width=bar_width, label=model)
plt.xticks(base_positions + (len(scores.index) - 1) / 2 * bar_width, scores.columns)
plt.ylabel("Perfomance gap, %", fontsize=18)
plt.xlabel("Dataset", fontsize=18)
plt.ylim(0, 7)
plt.annotate("10%", xy=(1.5, 7), xytext=(1.2, 6), arrowprops=dict(facecolor='steelblue', shrink=0.03))
'''
for base_position, dataset in zip(base_positions, scores.columns):
    left, right = plt.xlim()
    span = right - left
    plt.axhline(y=greedy_treshold[dataset],
        xmin=(base_position - 0.5 * bar_width - left) / span,
        xmax=(base_position + 2.5 * bar_width - left) / span,
        linestyle="dashed",
        color="red",
        label="Greedy" if base_position==0 else None)
'''
plt.legend()
