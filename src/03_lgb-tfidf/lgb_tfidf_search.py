# _*_ coding: utf-8 _*_

"""
Tune the hyper-parameters of LightGBM model, where
the feature vectors are extracted by using TF-IDF algorithm.

Author: StrongXGP
Date:	2018/07/16
"""

import gc
import datetime
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from time import time
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score

warnings.filterwarnings('ignore')

# Load data
# ============================================================================

print("Load data...")
data_path = "../../raw_data/train_set.csv"
data = pd.read_csv(data_path)

X_text = data['word_seg']
y = data['class'] - 1
num_classes = max(y) + 1
print("The number of samples is: %d" % len(X_text))
print("The number of classes is: %d" % num_classes)

del data
gc.collect()

# Tuning the hyper-parameters of LightGBM model and save the results
# ============================================================================

df_params = pd.read_csv("lgb-tfidf-params.csv")
num_params = df_params.shape[0]
print()
print("The number of parameter combinations is: %d" % num_params)

for i in range(num_params):
    print()
    print("Parameter combination %d" % (i + 1))

    # Extractor's hyper-parameters
    vect_params = {
        'ngram_range': tuple(map(int, df_params['ngram_range'].values[i].split('|'))),
        'max_df': df_params['max_df'].values[i],
        'min_df': df_params['min_df'].values[i],
        'max_features': None if df_params['max_features'].values[i] == 'None' else int(df_params['max_features'].values[i]),
        'sublinear_tf': df_params['sublinear'].values[i]
    }
    print("Vectorizer's hyper-parameters:")
    pprint(vect_params)

    # Initialize feature extractor
    vectorizer = TfidfVectorizer(**vect_params)

    print("Extract features...")
    t0_extract = time()
    X = vectorizer.fit_transform(X_text)
    print("Done in %.3f seconds." % (time() - t0_extract))
    print("Extract finished! ( ^ _ ^ ) V")

    # The number of features
    num_feats = len(vectorizer.get_feature_names())
    print("The number of features is %d" % num_feats)

    print("Split data into training and validation set...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    gbm_params = {
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
    print("LightGBM's hyper-parameters:")
    pprint(gbm_params)

    print("Start training...")
    t0_train = time()
    evals_result = {}
    gbm = lgb.train(params=gbm_params,
                    train_set=lgb_train,
                    num_boost_round=5000,
                    valid_sets=[lgb_train, lgb_val],
                    valid_names=['train', 'val'],
                    evals_result=evals_result,
                    early_stopping_rounds=50,
                    verbose_eval=100)
    print("Done in %.3f seconds" % (time() - t0_extract))
    print("Training finished! ( ^ _ ^ ) V")

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

    res = "%s,%s,%d,%s,%f,%d,%s,%s,%s,%.4f,%d,%d,%d,%.4f,%.4f,%d,%.4e,%.4e,%.4e,%.4e,%.4e,%d,%.5f,%.5f,%.2f,%.2f,%.5f,%.5f\n" % (
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "lgb-tfidf-%d" % (i + 1),  # model name
        num_feats,  # number of features
        df_params['ngram_range'].values[i],
        vect_params['max_df'],
        vect_params['min_df'],
        vect_params['max_features'],
        vect_params['sublinear_tf'],
        gbm_params['boosting_type'],
        gbm_params['learning_rate'],
        gbm_params['num_leaves'],
        gbm_params['max_depth'],
        gbm_params['min_data_in_leaf'],
        gbm_params['feature_fraction'],
        gbm_params['bagging_fraction'],
        gbm_params['bagging_freq'],
        gbm_params['lambda_l1'],
        gbm_params['lambda_l2'],
        gbm_params['min_gain_to_split'],
        gbm_params['min_sum_hessian_in_leaf'],
        0.8,  # train size
        best_iter,  # best iteration
        loss_train,  # multi-logloss of training set
        loss_val,  # multi-logloss of validation set
        acc_train,  # accuracy of training set
        acc_val,  # accuracy of validation set
        f1_train,  # f1 score of training set
        f1_val  # f1 score of validation set
    )

    f = open("lgb-tfidf-tuning-results.csv", 'a')
    f.write(res)
    f.close()
