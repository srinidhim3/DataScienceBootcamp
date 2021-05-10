import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv('data/Country clusters standardized.csv', index_col='Country')

x_scaled = data.drop(['Language'], axis=1)

sns.clustermap(x_scaled, cmap='mako')
plt.savefig('graphs/heatmap.png')