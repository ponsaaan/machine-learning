from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

svm = SVC(C=100)
svm.fit(X_train, y_train)
print(f'テストセットの正解率: {svm.score(X_test, y_test)}')

# 0-1スケール変換で前処理
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 変換されたデータで学習
svm.fit(X_train_scaled, y_train)
print(f'スケール変換後のテストセットの正解率: {svm.score(X_test_scaled, y_test)}')

# 平均0分散１で前処理
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)
print(f'正規化したデータセットでのテストセット正解率: {svm.score(X_test_scaled, y_test)}')

