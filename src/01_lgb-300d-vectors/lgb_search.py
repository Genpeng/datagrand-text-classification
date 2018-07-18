# _*_ coding: utf-8 _*_

"""
Tune the hyper-parameters of LightGBM model.

Author: StrongXGP
Date:	2018/07/16
"""

import gc
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# Load data
# ============================================================================

print("Load data...")
data_path = "../../processed_data/train-data-word-300d-mean.txt"
data = pd.read_csv(data_path)

X = data.drop(['class'], axis=1)
y = data['class'] - 1
num_feats = X.shape[1]
num_classes = max(y) + 1

print("Split data into training and validation set...")
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)

del data, X, y
gc.collect()

# Tuning the hyper-parameters of LightGBM model and save the results
# ============================================================================

lgb_train = lgb.Dataset(X_train.values, y_train.values)
lgb_val = lgb.Dataset(X_val.values, y_val.values, reference=lgb_train)

df_params = pd.read_csv("lgb-word-300d-mean-params.csv")
num_params = df_params.shape[0]
for i in range(num_params):
    params = {
        'boosting_type': df_params['type'].values[i],
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',

        'learning_rate': df_params['lr'].values[i],

        'num_leaves': df_params['n_leaf'].values[i],
        'max_depth': df_params['n_depth'].values[i],
        'min_data_in_leaf': df_params['min_data'].values[i],

        'feature_fraction': df_params['feat_frac'].values[i],
        'bagging_fraction': df_params['bagging_frac'].values[i],
        'bagging_freq': df_params['bagging_freq'].values[i],

        'lambda_l1': df_params['l1'].values[i],
        'lambda_l2': df_params['l2'].values[i],
        'min_gain_to_split': df_params['min_gain'].values[i],
        'min_sum_hessian_in_leaf': df_params['hessian'].values[i],

        'num_threads': 16,
        'verbose': 0,
        'is_training_metric': 'True'
    }

    print("Hyper-parameters:")
    print(params)

    print("Start training...")
    evals_result = {}
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=5000,
                    valid_sets=[lgb_train, lgb_val],
                    valid_names=['train', 'val'],
                    evals_result=evals_result,
                    early_stopping_rounds=50,
                    verbose_eval=100)
    print("Training finished! ^_^")

    best_iter = gbm.best_iteration
    loss_train = evals_result['train']['multi_logloss'][best_iter-1]
    loss_val = evals_result['val']['multi_logloss'][best_iter-1]

    probs_train = gbm.predict(X_train, num_iteration=best_iter)
    preds_train = np.argmax(probs_train, axis=1)
    acc_train = accuracy_score(y_train, preds_train)
    f1_train = f1_score(y_train, preds_train, average='weighted')

    probs_val = gbm.predict(X_val, num_iteration=best_iter)
    preds_val = np.argmax(probs_val, axis=1)
    acc_val = accuracy_score(y_val, preds_val)
    f1_val = f1_score(y_val, preds_val, average='weighted')

    print("Best iteration: %d" % best_iter)
    print("Training Loss: %.5f, Validation Loss: %.5f" % (loss_train, loss_val))
    print("Training Accuracy: %.2f, Validation Accuracy: %.2f" % (acc_train * 100, acc_val * 100))
    print("Training F1 Score: %.5f, Validation F1 Score: %.5f" % (f1_train, f1_val))

    res = "%s,%s,%d,%s,%.4f,%d,%d,%d,%.4f,%.4f,%d,%.4e,%.4e,%.4e,%.4e,%.4e,%d,%.5f,%.5f,%.2f,%.2f,%.5f,%.5f\n" % (
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "lgb-300d-sum",  # model name
        num_feats,  # number of features
        params['boosting_type'],
        params['learning_rate'],
        params['num_leaves'],
        params['max_depth'],
        params['min_data_in_leaf'],
        params['feature_fraction'],
        params['bagging_fraction'],
        params['bagging_freq'],
        params['lambda_l1'],
        params['lambda_l2'],
        params['min_gain_to_split'],
        params['min_sum_hessian_in_leaf'],
        0.8,  # train size
        best_iter,  # best iteration
        loss_train,  # multi-logloss of training set
        loss_val,  # multi-logloss of validation set
        acc_train,  # accuracy of training set
        acc_val,  # accuracy of validation set
        f1_train,  # f1 score of training set
        f1_val  # f1 score of validation set
    )

    f = open("lgb-word-300d-mean-tuning-results.csv", 'a')
    f.write(res)
    f.close()
