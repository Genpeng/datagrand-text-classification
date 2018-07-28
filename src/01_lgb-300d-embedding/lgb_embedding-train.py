# _*_ coding: utf-8 _*_

"""
Train LightGBM model.

Author: StrongXGP
Date:	2018/07/16
"""

import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score

# Load data
# ============================================================================

print("Load data...")
train_data_file = "../../processed_data/train-data-word-300d-mean.txt"
test_data_file = "../../processed_data/test-data-word-300d-mean.txt"
df_train = pd.read_csv(train_data_file)
df_test = pd.read_csv(test_data_file)

X_train = df_train.drop(['class'], axis=1)
y_train = df_train['class'] - 1
X_test = df_test
num_feats = X_train.shape[1]
num_classes = max(y_train) + 1

del df_train, df_test
gc.collect()

# Train the LightGBM model and make submission
# ============================================================================

lgb_train = lgb.Dataset(X_train.values, y_train.values)

df_params = pd.read_csv("lgb-word-300d-mean-tuning-results.csv").sort_values(by='f1_val', ascending=False)
for i in range(1):
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

    num_epochs = df_params['best_iter'].values[i]
    print("Round number: %d" % num_epochs)

    print("Start training...")
    evals_result = {}
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=num_epochs,
                    valid_sets=lgb_train,
                    valid_names=['train'],
                    evals_result=evals_result,
                    verbose_eval=100)
    print("Training finished! ^_^")

    best_iter = gbm.best_iteration
    loss_train = evals_result['train']['multi_logloss'][best_iter-1]

    probs_train = gbm.predict(X_train, num_iteration=best_iter)
    preds_train = np.argmax(probs_train, axis=1)
    acc_train = accuracy_score(y_train, preds_train)
    f1_train = f1_score(y_train, preds_train, average='weighted')

    print("Best round: %d" % best_iter)
    print("Training Loss: %.5f" % loss_train)
    print("Training Accuracy: %.2f" % (acc_train * 100))
    print("Training F1 Score: %.5f" % f1_train)

    feature_importance = pd.DataFrame({'name': gbm.feature_name(), 'importance': gbm.feature_importance()})
    feature_importance.sort_values(by='importance', ascending=False, inplace=True)
    feature_importance.to_csv("feature-importance-char-300d-mean.csv", index=False)

    # Make submission
    probs_test = gbm.predict(X_test)
    preds_test = np.argmax(probs_test, axis=1) + 1
    df_test = pd.read_csv("../../raw_data/test_set.csv")
    submission = pd.DataFrame()
    submission['id'] = df_test['id']
    submission['class'] = preds_test
    submission.to_csv("2018-07-17_lgb-word-300d-mean-submission.csv", index=False)
