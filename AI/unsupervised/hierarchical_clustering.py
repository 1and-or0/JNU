import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv("/kaggle/input/customer-personality-analysis/marketing_campaign.csv", sep="\t")
df.columns = df.columns.str.strip()  # 모든 열 이름에서 앞뒤 공백 제거

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
df.info()

# 정규화
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 덴드로그램 시각화
plt.figure(figsize=(12, 6))
dendrogram(linked,
        truncate_mode='level',  # 레벨 단위로 자름 (전체 노드가 많을 경우)
        p=30,                  # 보여줄 최대 레벨 수
        leaf_rotation=90.,
        leaf_font_size=8.,
        color_threshold=0.7*max(linked[:, 2]))  # 클러스터 분할 기준선

plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('Sample index or (cluster size)')
plt.ylabel('Distance')
plt.grid(True)
plt.show()

# Hierarchical Clustering (Agglomerative)
k = 3
agg_clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
labels = agg_clustering.fit_predict(df_scaled)

# 차원 축소 (PCA 2D)
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.title(f"Hierarchical Clustering (k={k}) - PCA Projection")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label="Cluster Label")
plt.grid(True)
plt.show()
