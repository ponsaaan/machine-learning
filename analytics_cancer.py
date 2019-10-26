from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from mglearn import discrete_scatter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA
# データの最初の２つの主成分だけ維持する
pca = PCA(n_components=2)
# cancerデータセットにPCAモデルを適合
pca.fit(X_scaled)

# 最初の２つの主成分に対してデータポイントを変換
X_pca = pca.transform(X_scaled)
print(f'元のshape: {str(X_scaled.shape)}')
print(f'PCA後のshape: {str(X_pca.shape)}')

# 第１主成分と第２主成分によるプロット
plt.figure(figsize=(8, 8))
discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("第１主成分")
plt.ylabel("第２主成分")

plt.show()

# 主成分を構成する特徴量
print(f'PC　component shape: {pca.components_}')
