"""
PassengerId
Survived
Pclass（チケットの等級）
Name
Sex
Age
SibSp（兄弟の数）
Parch（親の数）
Ticket（チケット名）
Fare（運賃）
Cabin
Embarked（乗船港）

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import linear_model

# 1. データセットの読み込み
pd.set_option('display.max_column', 100)
pd.set_option('display.max_rows', 100)
train_df = pd.read_csv('datasets/titanic/train.csv')
test_df = pd.read_csv('datasets/titanic/test.csv')
print(f'訓練データの個数： {train_df.shape[0]}.')
print(f'テストセットの個数： {test_df.shape[0]}.')

# 2. データの特徴や欠損値を評価
print(f'訓練セットの欠損値数： {train_df.isnull().sum()}')
print(f'テストセットの欠損値数： {test_df.isnull().sum()}')
# result: Age（19.9%）,Cabin（77.1%）,Embarked（0.2%）に欠損値あり

# 2-1. 年齢についてのヒストグラム
# ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
# train_df["Age"].plot(kind='density', color='teal')
# ax.set(xlabel='Age')
# plt.xlim(-10, 85)
# plt.show()

# 2-2. 年齢の中央値を計算
mean_age = train_df['Age'].mean(skipna=True)
print(f'年齢の中央値：{mean_age}')

# 2-3. 乗船港についてのヒストグラム
# print(train_df['Embarked'].value_counts())
# sns.countplot(x='Embarked', data=train_df, palette='Set2')
# plt.show()

# 2-4. Age,Cabin,Embarkedについての方針
# 年齢がNaNの箇所は中央値で埋める
# Cabinはデータとして無視する
# EmbarkedがNaNの箇所はSで埋める
# EmbarkedはS=> 0, Q=> 1, C=> 2に変換
# テストセットの運賃が一つだけNaNであるため、中央値で埋める
train_df["Age"].fillna(mean_age, inplace=True)
train_df["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)
train_df['Embarked'] = train_df["Embarked"].replace('S', 0).replace('Q', 1).replace('C', 2)
train_df.drop('Cabin', axis=1, inplace=True)
test_df["Age"].fillna(mean_age, inplace=True)
test_df["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)
test_df['Embarked'] = test_df["Embarked"].replace('S', 0).replace('Q', 1).replace('C', 2)
test_df.drop('Cabin', axis=1, inplace=True)
test_df['Fare'].fillna(train_df['Fare'].mean(skipna=True), inplace=True)
print(f'訓練セットの欠損値の数: {train_df.isnull().sum()}')
print(f'テストセットの欠損値の数: {test_df.isnull().sum()}')

# 2-5. 家族についての考察
# 家族もしくは兄弟がいれば1人で来ている（0）、それ以外ならば家族で来ている（1）
train_df['TravelAlone'] = np.where((train_df["SibSp"] + train_df["Parch"]) > 0, 0, 1)
train_df.drop('SibSp', axis=1, inplace=True)
train_df.drop('Parch', axis=1, inplace=True)
test_df['TravelAlone'] = np.where((test_df["SibSp"] + test_df["Parch"]) > 0, 0, 1)
test_df.drop('SibSp', axis=1, inplace=True)
test_df.drop('Parch', axis=1, inplace=True)

# 2-6. 性別を1,0に変換（女性：1, 男性：0）
train_df = train_df.replace('male', 0).replace('female', 1)
test_df = test_df.replace('male', 0).replace('female', 1)

# 2-7. 乗客ID, 名前, Ticketは落とす
train_df.drop('PassengerId', axis=1, inplace=True)
train_df.drop('Name', axis=1, inplace=True)
train_df.drop('Ticket', axis=1, inplace=True)
test_df.drop('PassengerId', axis=1, inplace=True)
test_df.drop('Name', axis=1, inplace=True)
test_df.drop('Ticket', axis=1, inplace=True)

print(f'データの前処理後の訓練セット: \n'
      f' {train_df}')
print(f'データの前処理後のテストセット: \n'
      f'{test_df}')

# 3-1. 年齢と生死の関係についての考察
plt.figure(figsize=(15, 8))
ax = sns.kdeplot(train_df["Age"][train_df.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(train_df["Age"][train_df.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
ax.set(xlabel='Age')
plt.xlim(0, 85, 1)
plt.show()
# result: 年齢の相関はほとんどないが、子供が生き残る傾向にあるようだ
# １６歳未満で新しいカラムを追加
train_df['isChild'] = np.where(train_df['Age'] <= 16, 1, 0)
test_df['isChild'] = np.where(test_df['Age'] <= 16, 1, 0)

# 3-2. 運賃と生死についての考察
plt.figure(figsize=(15, 8))
ax = sns.kdeplot(train_df["Fare"][train_df.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(train_df["Fare"][train_df.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Fare for Surviving Population and Deceased Population')
ax.set(xlabel='Fare')
plt.xlim(0, 200)
plt.show()
# result: 運賃が低い乗客は死亡率が高い

# 3-3. PClassについての考察
sns.barplot('Pclass', 'Survived', data=train_df, color="darkturquoise")
plt.show()
# result: 部屋の等級が高いほど生存率が高い

# 3-4. 乗船港についての考察
sns.barplot('Embarked', 'Survived', data=train_df, color="teal")
plt.show()
# result: Cの港の生存率が高い

# 3-5. 家族で来ているか、1人で来ているかと生存率
sns.barplot('TravelAlone', 'Survived', data=train_df, color="mediumturquoise")
plt.show()
# result: 1人で来ている方が死亡率が高い=> 男性？

# 3-6. 性別と生存率
sns.barplot('Sex', 'Survived', data=train_df, color="aquamarine")
plt.show()
# result: 女性が生存率高い


# ロジスティック回帰モデルを作成
# lr = linear_model.LogisticRegression(random_state=0)
# lr.fit(x_train_std, y_train)

# 正解率を算出
# print(lr.predict(x_test_std))
