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
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

def preprocessing_titanic(data_frame):
    # 1. 年齢の欠損値は中央値で埋める
    mean_age = data_frame['Age'].mean(skipna=True)
    data_frame["Age"].fillna(mean_age, inplace=True)

    # 2. 乗船港は一番多い値で埋めて、数値に変換する
    data_frame["Embarked"].fillna(data_frame['Embarked'].value_counts().idxmax(), inplace=True)
    data_frame['Embarked'] = data_frame["Embarked"].replace('S', 0).replace('Q', 1).replace('C', 2)

    # 3. Cabinは消去する
    data_frame.drop('Cabin', axis=1, inplace=True)

    # 4. 運賃は中央値で埋める
    mean_fare = data_frame['Fare'].mean(skipna=True)
    data_frame['Fare'].fillna(mean_fare, inplace=True)

    # 5. 家族についての考察
    # 家族もしくは兄弟がいれば1人で来ている（0）、それ以外ならば家族で来ている（1）
    data_frame['TravelAlone'] = np.where((data_frame["SibSp"] + data_frame["Parch"]) > 0, 0, 1)
    data_frame.drop('SibSp', axis=1, inplace=True)
    data_frame.drop('Parch', axis=1, inplace=True)

    # 6. 性別を1,0に変換（女性：1, 男性：0）
    data_frame = data_frame.replace('male', 0).replace('female', 1)

    # 7. 乗客ID, 名前, Ticketは落とす
    data_frame.drop('PassengerId', axis=1, inplace=True)
    data_frame.drop('Name', axis=1, inplace=True)
    data_frame.drop('Ticket', axis=1, inplace=True)

    return data_frame


def consider_age(train_df):
    plt.figure(figsize=(15, 8))
    ax = sns.kdeplot(train_df["Age"][train_df.Survived == 1], color="darkturquoise", shade=True)
    sns.kdeplot(train_df["Age"][train_df.Survived == 0], color="lightcoral", shade=True)
    plt.legend(['Survived', 'Died'])
    plt.title('Density Plot of Age for Surviving Population and Deceased Population')
    ax.set(xlabel='Age')
    plt.xlim(0, 85, 1)
    plt.show()


def consider_fare(train_df):
    plt.figure(figsize=(15, 8))
    ax = sns.kdeplot(train_df["Fare"][train_df.Survived == 1], color="darkturquoise", shade=True)
    sns.kdeplot(train_df["Fare"][train_df.Survived == 0], color="lightcoral", shade=True)
    plt.legend(['Survived', 'Died'])
    plt.title('Density Plot of Fare for Surviving Population and Deceased Population')
    ax.set(xlabel='Fare')
    plt.xlim(0, 200)
    plt.show()


def consider_pclass(train_df):
    sns.barplot('Pclass', 'Survived', data=train_df, color="darkturquoise")
    plt.show()


def consider_embark(train_df):
    sns.barplot('Embarked', 'Survived', data=train_df, color="teal")
    plt.show()


def consider_travel_alone(train_df):
    sns.barplot('TravelAlone', 'Survived', data=train_df, color="mediumturquoise")
    plt.show()


def consider_sex(train_df):
    sns.barplot('Sex', 'Survived', data=train_df, color="aquamarine")
    plt.show()


def prepare_train_xy(train_df):
    train_X = train_df.values[:, 1:]
    train_y = train_df.values[:, 0]

    return train_X, train_y


def prepare_test_x(test_df):
    test_x = test_df.values[:, :]

    return test_x


# 1. データセットの読み込み
pd.set_option('display.max_column', 100)
pd.set_option('display.max_rows', 100)
train_df_all = pd.read_csv('datasets/titanic/train.csv')
test_df_all = pd.read_csv('datasets/titanic/test.csv')
print(f'訓練データの個数： {train_df_all.shape[0]}.')
print(f'テストセットの個数： {test_df_all.shape[0]}.')

# 2. データの特徴や欠損値を評価
print(f'訓練セットの欠損値数： {train_df_all.isnull().sum()}')
print(f'テストセットの欠損値数： {test_df_all.isnull().sum()}')
# result: Age（19.9%）,Cabin（77.1%）,Embarked（0.2%）に欠損値あり

# 2-1. 年齢についてのヒストグラム
# ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
# train_df["Age"].plot(kind='density', color='teal')
# ax.set(xlabel='Age')
# plt.xlim(-10, 85)
# plt.show()

# 2-2. 乗船港についてのヒストグラム
# print(train_df['Embarked'].value_counts())
# sns.countplot(x='Embarked', data=train_df, palette='Set2')
# plt.show()

# 2-3. Age,Cabin,Embarkedについての方針
# 年齢がNaNの箇所は中央値で埋める
# Cabinはデータとして無視する
# EmbarkedがNaNの箇所はSで埋める
# EmbarkedはS=> 0, Q=> 1, C=> 2に変換
# テストセットの運賃が一つだけNaNであるため、中央値で埋める
train_df = preprocessing_titanic(train_df_all)
test_df = preprocessing_titanic(test_df_all)

print(f'訓練セットの欠損値の数: {train_df.isnull().sum()}')
print(f'テストセットの欠損値の数: {test_df.isnull().sum()}')
print(f'データの前処理後の訓練セット: \n'
      f' {train_df}')
print(f'データの前処理後のテストセット: \n'
      f'{test_df}')


# # 3-1. 年齢と生死の関係についての考察
# consider_age(train_df)
# # result: 年齢の相関はほとんどないが、子供が生き残る傾向にあるようだ
#
# # 3-2. 運賃と生死についての考察
# consider_fare(train_df)
# # result: 運賃が低い乗客は死亡率が高い
#
# # 3-3. PClassについての考察
# consider_pclass(train_df)
# # result: 部屋の等級が高いほど生存率が高い
#
# # 3-4. 乗船港についての考察
# consider_embark(train_df)
# # result: Cの港の生存率が高い
#
# # 3-5. 家族で来ているか、1人で来ているかと生存率
# consider_travel_alone(train_df)
# # result: 1人で来ている方が死亡率が高い=> 男性？
#
# # 3-6. 性別と生存率
# consider_sex(train_df)
# # result: 女性が生存率高い

# 機械学習用の訓練セットとラベルを用意
train_X, train_y = prepare_train_xy(train_df)
# 評価用のテストセットを用意
test_x = prepare_test_x(test_df)

# 機械学習モデルで学習
# グリッドサーチで最適なパラメータを算出、、、と思ったけどテストセットの正解どこにもないじゃん...
# grid_params = [{
#     'n_estimators': [100, 500, 1000, 2000, 5000],
#     'max_depth': [1, 3, 5, 7, 9]
# }]
# gs = model_selection.grid_search.GridSearchCV()

# ランダムフォレスト
# random_forest = ensemble.RandomForestClassifier(n_estimators=5000, max_depth=6, random_state=0)
# clf_result = random_forest.fit(train_X, train_y)

# 勾配ブースティング
forest = ensemble.GradientBoostingClassifier(n_estimators=1000, random_state=0)
clf_result = forest.fit(train_X, train_y)

# 予測結果の算出
predict_survived_list = clf_result.predict(test_x).astype(int)

# 学習モデルの評価
# 訓練セットの正解率を算出
train_survived_list = clf_result.predict(train_X)
ac_score = metrics.accuracy_score(train_y, train_survived_list)
print(f'正解率：{ac_score}')

# 提出用のデータを用意
submission_df = test_df_all
print(test_df_all)
print(test_df_all.shape)
submission_df['Survived'] = predict_survived_list
print(submission_df[['PassengerId', 'Survived']])
# 予測結果のCSV出力
submission_df[['PassengerId', 'Survived']].to_csv('result_predicted/submission.csv', index=False)
