from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)

print(X_train.shape)
print(X_test.shape)

scaler = MinMaxScaler()
scaler.fit(X_train)

# 訓練データを変換
X_train_scaled = scaler.transform(X_train)
# スケール変換の前後のデータ特性をプリント
print(f'変換後のshape: {X_train_scaled.shape}')
print(f'変換前の最小値: {X_train.min(axis=0)}')
print(f'変換前の最大値: {X_train.max (axis=0)}')
print(f'変換後の最小値: {X_train_scaled.min(axis=0)}')
print(f'変換後の最大値: {X_train_scaled.max(axis=0)}')

# テストデータを変換
X_test_scaled = scaler.transform(X_test)
# スケール変換の前後のデータ特性をプリント
print(f'スケール変換後の最小値: {X_test_scaled.min(axis=0)}')
print(f'スケール変換後の最大値: {X_test_scaled.max(axis=0)}')

