import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('data/3.12. Example.csv')

plt.scatter(data['Satisfaction'], data['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.savefig('graphs/Market_segmentation_1.png')

x = data.copy()
kmeans = KMeans(2)
kmeans.fit(x)

cluster = x.copy()
cluster['cluster_pred'] = kmeans.fit_predict(x)

plt.clf()
plt.scatter(cluster['Satisfaction'], cluster['Loyalty'], c=cluster['cluster_pred'], cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.savefig('graphs/Market_segmentation_2.png')

from sklearn import  preprocessing
x_scaled = preprocessing.scale(x)

wcss = []
for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

plt.clf()
plt.plot(range(1,10), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('graphs/Market_segmentation_3.png')

kmeans_new = KMeans(4)
kmeans_new.fit(x_scaled)
cluster_new = x.copy()
cluster_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)

plt.clf()
plt.scatter(cluster_new['Satisfaction'], cluster_new['Loyalty'], c=cluster_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.savefig('graphs/Market_segmentation_4.png')