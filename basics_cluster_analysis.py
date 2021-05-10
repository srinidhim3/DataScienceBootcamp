import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from  sklearn.cluster import KMeans

data = pd.read_csv('data/3.01. Country clusters.csv')

plt.scatter(data['Longitude'],data['Latitude'])
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.savefig('graphs/basics_cluster_analysis_1.png')

x = data.iloc[:,1:3]

kmeans = KMeans(3)
kmeans.fit(x)

identified_cluster = kmeans.fit_predict(x)
print(identified_cluster)

data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identified_cluster

plt.scatter(data_with_clusters['Longitude'],data_with_clusters['Latitude'], c=data_with_clusters['Cluster'],cmap='rainbow')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.savefig('graphs/basics_cluster_analysis_2.png')

# Elbow method
wcss = []
for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
print(wcss)

plt.clf()
plt.plot(range(1,7), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within clusters sum of squares')
plt.savefig('graphs/basics_cluster_analysis_3.png')