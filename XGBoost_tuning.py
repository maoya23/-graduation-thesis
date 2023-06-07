import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
import time
from sklearn.metrics import accuracy_score

df=pd.read_csv('data/alzheimer_full.csv',index_col='IDENTIFIER')

#df=df.drop(columns='GeneName')
train_set,test_set=train_test_split(df,test_size=0.2,random_state=4)#trainとtestを1:4の割合で分割する、random_stateを固定することでいつも同じ抽出方法

X_train=train_set.drop('Label',axis=1)
y_train=train_set['Label']#Labelをもとに判定したい、訓練セット

X_test=test_set.drop('Label',axis=1)
y_test=test_set['Label']

#X_train=X_train.values
#y=y_train.values

dtrain=xgb.DMatrix(X_train,label=y_train)#xgbが判定できるような特別な配列に変換
dtest=xgb.DMatrix(X_test,label=y_test)


seed=42
model = XGBClassifier(booster='gbtree', objective='binary:logistic',random_state=seed, n_estimators=10000)  # チューニング前のモデル,2値分類なので、objectiveはbinary:logistic

cv = KFold(n_splits=5, shuffle=True, random_state=seed)  # KFoldでクロスバリデーション分割指定

fit_params = {'verbose': 0,  # 学習中のコマンドライン出力
'early_stopping_rounds': 10,  # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
'eval_metric': 'rmse',  # early_stopping_roundsの評価指標
'eval_set': [(X_train, y_train)]  # early_stopping_roundsの評価指標算出用データ
}

scoring='neg_log_loss'#2値分類の一般的な評価指標はlogloss


cv_params = {'subsample': [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
            'reg_lambda': [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
            'learning_rate': [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
            'min_child_weight': [1, 3, 5, 7, 9, 11, 13, 15],
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'gamma': [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
            }
param_scales = {'subsample': 'linear',
                'colsample_bytree': 'linear',
                'reg_alpha': 'log',
                'reg_lambda': 'log',
                'learning_rate': 'log',
                'min_child_weight': 'linear',
                'max_depth': 'linear',
                'gamma': 'log'
                }
# 検証曲線のプロット（パラメータ毎にプロット）
for i, (k, v) in enumerate(cv_params.items()):
    train_scores, valid_scores = validation_curve(estimator=model,
                                                X=X_train, y=y_train,
                                                param_name=k,
                                                param_range=v,
                                                fit_params=fit_params,
                                                cv=cv, scoring=scoring,
                                                n_jobs=64)
    # 学習データに対するスコアの平均±標準偏差を算出
    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    train_center = train_mean
    train_high = train_mean + train_std
    train_low = train_mean - train_std
    # テストデータに対するスコアの平均±標準偏差を算出
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std  = np.std(valid_scores, axis=1)
    valid_center = valid_mean
    valid_high = valid_mean + valid_std
    valid_low = valid_mean - valid_std
    # training_scoresをプロット
    fig = plt.figure()
    plt.plot(v, train_center, color='blue', marker='o', markersize=5, label='training score')
    plt.fill_between(v, train_high, train_low, alpha=0.15, color='blue')
    # validation_scoresをプロット
    plt.plot(v, valid_center, color='green', linestyle='--', marker='o', markersize=5, label='validation score')
    plt.fill_between(v, valid_high, valid_low, alpha=0.15, color='green')
    # スケールをparam_scalesに合わせて変更
    plt.xscale(param_scales[k])
    # 軸ラベルおよび凡例の指定
    plt.xlabel(k)  # パラメータ名を横軸ラベルに
    plt.ylabel(scoring)  # スコア名を縦軸ラベルに
    plt.legend(loc='lower right')  # 凡例
    # グラフを描画
    plt.show()
    fig.savefig("graph/alzheimer_full%03.f"%(i)+".png")
