import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

dfcsv = pd.read_csv("csv/Online_Shopping_Behavior.csv")
dfcsv.index = dfcsv.index + 1
df = dfcsv.rename(columns={"Session_Duration": "dur",
                           "Page_Views": "view",
                           "Purchase_Amount": "amount",
                           "Bounce_Rate": "brate"})

columns = ["dur", "view", "amount", "brate"]

# print(df.info())
# column - null count - datatype
# dur - 0 - float64
# view - 600 - float64              // Need to handle missing values in this column
# amount - 0 - float64
# brate - 0 - float64

df_kn = df.copy()
df_kn[columns] = KNNImputer(n_neighbors=5).fit_transform(df[columns])

print(df.info())                    # // view : 5400 data
print(df_kn.info())                 # // view : 6000 data

df_std = df_kn.copy()
df_std = StandardScaler().fit_transform(df_std[columns])

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_std)
    wcss.append(kmeans.inertia_)

print(wcss)
# OUTPUT:
# 24000.0, 19514.238279859466, 
# 16744.673090000808, 14566.596130299149, 
# 12932.877937525875, 11710.686953046728, 
# 10558.951293487724, 9477.004120885456, 
# 8980.541076626609, 8448.16194430772

wdiff = []
for i in range(1, len(wcss)):
    x = wcss[i-1] - wcss[i]
    wdiff.append(x)

print(wdiff)
# OUTPUT: 
# 4485.761720140534, 2769.5651898586584,
# 2178.076959701659, 1633.7181927732745,
# 1222.1909844791462, 1151.7356595590045,
# 1081.9471726022675, 496.46304425884773, 532.3791323188889

# Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method: Optimal k', fontsize=14)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('WCSS', fontsize=12)
plt.xticks(range(1, 11))
plt.grid()
plt.show()

# K Means Plot
kmeans = KMeans(n_clusters=5, random_state=42)
df_kn["KN-Cluster"] = kmeans.fit_predict(df_std)
print(df_kn["KN-Cluster"].value_counts())
cluster_summary = df_kn.groupby("KN-Cluster").mean()
print(cluster_summary)
print(df_kn.head())

# DBSCAN Plot
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_std)
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df_kn["KN-Cluster"], cmap="viridis", alpha=0.7)
plt.title("KNear Clustering with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()

dbscan = DBSCAN(eps=0.5, min_samples=5)
df_kn['DBSCAN-Cluster'] = dbscan.fit_predict(df_std)
print("Unique clusters:", np.unique(dbscan.fit_predict(df_std)))
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_std)
plt.figure(figsize=(8, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=dbscan.fit_predict(df_std), cmap='viridis', s=10)
plt.title("DBSCAN Clustering with PCA", fontsize=14)
plt.xlabel("PCA Component 1", fontsize=12 )
plt.ylabel("PCA Component 2", fontsize=12)
plt.colorbar(label="Cluster")
plt.show()