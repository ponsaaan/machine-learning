from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people


people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

fix, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})

for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])

print(f"クラスの数: {people.target_names}")


# 該当データセットは偏っている（ブッシュとバウエルの画像が多い）
counts = np.bincount(people.target)
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print(name, count)
    if (i + 1) % 3 == 0:
        print()

# 各人の画像数を50に制限する
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = True

X_people = people.data[mask] / 255
y_people = people.target[mask]


# K近傍法を適用してみる
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print(f'スコア: {knn.score(X_test, y_test)}')

# pcaを適用する
from sklearn.decomposition import PCA

pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(f'主成分分析後の訓練セット形状: {X_train_pca.shape}')

# PCA後の訓練セットで学習させてみる
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print(f'PCA後のスコア: {knn.score(X_test_pca, y_test)}')


