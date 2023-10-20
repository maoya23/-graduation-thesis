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
import analysis.XGBoost_tuning as XGBoost_tuning


model=XGBoost_tuning.model
X_train=XGBoost_tuning.X_train
y_train=XGBoost_tuning.y_train
cv=XGBoost_tuning.cv
scoring=XGBoost_tuning.scoring
fit_params=XGBoost_tuning.fit_params

start = time.time()
# ベイズ最適化時の評価指標算出メソッド
def bayes_objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.8),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0001, 0.01, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1, log=True),
        'gamma': trial.suggest_float('gamma', 0.001, 0.01, log=True),
    }
    # モデルにパラメータ適用
    model.set_params(**params)
    # cross_val_scoreでクロスバリデーション
    scores = cross_val_score(model, X_train, y_train, cv=cv,
                            scoring=scoring, fit_params=fit_params, n_jobs=64)
    val = scores.mean()
    return val

# ベイズ最適化を実行
study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(bayes_objective, n_trials=600)

# 最適パラメータの表示と保持
best_params = study.best_trial.params
best_score = study.best_trial.value
print(f'最適パラメータ {best_params}\nスコア {best_score}')
print(f'所要時間{time.time() - start}秒')
