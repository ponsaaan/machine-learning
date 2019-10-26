from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = make_blobs(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

print(f'テストセットのスコア: {logreg.score(X_test, y_test)}')

# 交差検証
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression()

scores = cross_val_score(logreg, iris.data, iris.target, cv=10)
print(f'交差検証スコア: {scores}')
print(f'交差検証平均スコア: {scores.mean()}')

# グリッドサーチ
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

grid_search.fit(X_train, y_train)
print(f'テストセットのスコア: {grid_search.score(X_test, y_test)}')
print(f'ベストパラメータ: {grid_search.best_params_}')

import pandas as pd
pd.set_option('display.max_column', 100)
results = pd.DataFrame(grid_search.cv_results_)
print(results.head())


# カーネル

