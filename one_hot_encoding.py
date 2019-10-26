from sklearn import datasets
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mglearn
import os

pd.set_option('display.max_column', 100)

adult_path = os.path.join(mglearn.datasets.DATA_PATH, "adult.data")
data = pd.read_csv(
    adult_path, header=None, index_col=False,
    names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income']
)

data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
             'occupation', 'income']]

print(data.head())

print(data.gender.value_counts())

print(f'元の特徴量: {list(data.columns)}')
data_dummies = pd.get_dummies(data)
print(f'変換後の特徴量: {list(data_dummies.columns)}')

features = data_dummies.ix[:, 'age':'occupation_ Transport-moving']
X = features.values
y = data_dummies['income_ >50K'].values
print(f'X.shape: {X.shape}, y.shape: {y.shape}')

# 整数特徴量をダミー変数に変換する
demo_df = pd.DataFrame({
    'integer feature': [0, 1, 2, 1],
    'categorical feature': ['socks', 'fox', 'socks', 'box']
})
print(demo_df)
print(pd.get_dummies(demo_df))

demo_df['integer feature'] = demo_df['integer feature'].astype(str)
print(pd.get_dummies(demo_df, columns=['integer feature', 'categorical feature']))

