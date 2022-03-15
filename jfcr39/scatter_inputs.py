import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import colorsys
import random
import os
from matplotlib.mlab import PCA as mlabPCA
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA

pca = sklearnPCA(n_components=2)

X = pd.read_csv(
    r"C:\Users\Vaibhav\NMF\vaibhav_NMF\FinalNMF\input/input_small_data_DRUG.zero-one.csv",
    index_col=0,
    na_values="NaN",
)
X = X.fillna(0)

print("drugs")
X_norm = (X - X.min()) / (X.max() - X.min())
transformed = pca.fit_transform(X_norm)
kmeans = KMeans(n_clusters=3, random_state=0).fit(transformed)
x_kmeans = kmeans.predict(transformed)
plt.scatter(transformed[:, 0], transformed[:, 1], c=x_kmeans, s=50, cmap="viridis")
plt.title("Input drugs")
plt.show()
"""
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
"""
plt.show()

Y = pd.read_csv(
    r"C:\Users\Vaibhav\NMF\vaibhav_NMF\FinalNMF\input/input_small_data_MUT.zero-one.csv",
    index_col=0,
    na_values="NaN",
)
Y = Y.fillna(0)

print("muatations")
Y_norm = (Y - Y.min()) / (Y.max() - Y.min())
transformed = pca.fit_transform(Y_norm)
kmeans = KMeans(n_clusters=3, random_state=0).fit(transformed)
y_kmeans = kmeans.predict(transformed)
plt.scatter(transformed[:, 0], transformed[:, 1], c=y_kmeans, s=50, cmap="viridis")
plt.title("Input mutations")
plt.show()
"""
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
"""

Z = pd.read_csv(
    r"C:\Users\Vaibhav\NMF\vaibhav_NMF\FinalNMF\input/input_small_data_PROT.zero-one.csv",
    index_col=0,
    na_values="NaN",
    encoding="ISO-8859-1",
)
Z = Z.fillna(0)

print("proteins")
Z_norm = (Z - Z.min()) / (Z.max() - Z.min())
transformed = pca.fit_transform(Z_norm)
kmeans = KMeans(n_clusters=3, random_state=0).fit(transformed)
z_kmeans = kmeans.predict(transformed)
plt.scatter(transformed[:, 0], transformed[:, 1], c=z_kmeans, s=50, cmap="viridis")
plt.title("Input proteins")
plt.show()
"""
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
"""
plt.show()
