import numpy as np # linear algebra
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("/kaggle/input/customer-personality-analysis/marketing_campaign.csv", sep="\t")
df.columns = df.columns.str.strip()  # 모든 열 이름에서 앞뒤 공백 제거
# df.info()
to_use = [
    "Year_Birth",
    "Income",	
    "Recency",	
    "MntWines",	
    "MntFruits",	
    "MntMeatProducts",	
    "MntFishProducts",	
    "MntSweetProducts",	
    "MntGoldProds",	
    "NumDealsPurchases",	
    "NumWebPurchases",	
    "NumCatalogPurchases",	
    "NumStorePurchases",	
    "NumWebVisitsMonth"
]
df = df[to_use]
df = df.dropna()
print("Number of datapoints:", len(df))
# df.head()
df.info()

# 정규화
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

sse = []
K_range = range(1, 21)
for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    kmeans.fit(df_scaled)
    sse.append(kmeans.inertia_)  

plt.plot(K_range, sse, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("SSE (inertia)")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# KMeans 클러스터링 
k = 3
kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
kmeans.fit(df_scaled)
labels = kmeans.labels_

# 차원 축소 (PCA 2D)
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.title(f"KMeans Clustering (k={k}) - PCA Projection")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label="Cluster Label")
plt.grid(True)
plt.show()