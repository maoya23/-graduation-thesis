import analysis.XGBoost_tuning as XGBoost_tuning
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
from sklearn.metrics import roc_auc_score


model=XGBoost_tuning.model
X_train=XGBoost_tuning.X_train
y_train=XGBoost_tuning.y_train
cv=XGBoost_tuning.cv
scoring=XGBoost_tuning.scoring
fit_params=XGBoost_tuning.fit_params
dtrain=XGBoost_tuning.dtrain
dtest=XGBoost_tuning.dtest
y_test=XGBoost_tuning.y_test

xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.13802256787767156, 
        'min_child_weight': 1, 
        'max_depth': 1, 
        'colsample_bytree': 0.3786859191014443, 
        'subsample': 0.6767756571562955, 
        'reg_alpha': 0.0023292267134489906, 
        'reg_lambda': 0.0012830348663675152, 
        'gamma': 0.005100725978223754
        }


evals = [(dtrain, 'train'), (dtest, 'eval')]
evals_result = {}
bst = xgb.train(xgb_params,
                    dtrain,
                    num_boost_round=5000,
                    # 一定ラウンド回しても改善が見込めない場合は学習を打ち切る
                    early_stopping_rounds=10,
                    evals=evals,
                    evals_result=evals_result,
                    )

y_pred_proba = bst.predict(dtest)
y_pred = np.where(y_pred_proba > 0.5, 1, 0)
acc = roc_auc_score(y_test, y_pred)
print('Accuracy:', acc)


train_metric = evals_result['train']['logloss']
plt.plot(train_metric, label='train logloss')
eval_metric = evals_result['eval']['logloss']
plt.plot(eval_metric, label='eval logloss')
plt.grid()
plt.legend()
plt.xlabel('rounds')
plt.ylabel('logloss')
plt.savefig('graph/Alzheimer_curve_Brunner.png')

    # 性能向上に寄与する度合いで重要度をプロットする
git, ax = plt.subplots(figsize=(12, 12))
xgb.plot_importance(bst, height=0.8, ax=ax,importance_type='gain')
plt.show()
plt.savefig('importance_alzheimer.png')
