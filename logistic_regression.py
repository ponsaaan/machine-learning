from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn .linear_model import LogisticRegression

# irisのデータセットを読み込んでtrainとtestに分割
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)

# trainデータの平均と標準偏差を用いて標準化
sc = preprocessing.StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# ロジスティック回帰モデルで学習
lr = LogisticRegression(C=1000, random_state=0)
lr.fit(X_train_std, y_train)

# sentosa, versicolor, virginicaである確率を算出
print(lr.predict_proba(X_test_std[0, :].reshape(1, -1)))
