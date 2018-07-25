# _*_ coding: utf-8 _*_

"""
Test the fusion features (TF-IDF features + embedding features + topic model features)
where the prediction model is LightGBM.

Author: StrongXGP
Date:	2018/07/24
"""

import gc
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from time import time
from pprint import pprint
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Load data
# ============================================================================

print("Load data...")
data_path = "../../raw_data/train_demo.csv"
data = pd.read_csv(data_path)

X_text = data['word_seg']
y = data['class'] - 1
num_classes = max(y) + 1
print("The number of samples is: %d" % len(X_text))
print("The number of classes is: %d" % num_classes)

del data
gc.collect()

# Load character/word embedding
# ============================================================================

print("Load character and word embedding...")
char_embed_file = "../../processed_data/train-data-char-300d-mean.txt"
char_embed = pd.read_csv(char_embed_file).drop(['class'], axis=1)
word_embed_file = "../../processed_data/train-data-word-300d-mean.txt"
word_embed = pd.read_csv(word_embed_file).drop(['class'], axis=1)

# Extract TF features
# ============================================================================

tf_vect_params = {
    'ngram_range': (1, 2),
    'min_df': 3,
    'max_df': 0.9,
    'max_features': 20000,
}
tf_vectorizer = CountVectorizer(**tf_vect_params)
print("CountVectorizer's hyper-parameters:")
pprint(tf_vect_params)

print("Extract term frequency (TF) features...")
t0_extract = time()
X_tf = tf_vectorizer.fit_transform(X_text)
print("Done in %.3f seconds" % (time() - t0_extract))
print("Extract finished! ( ^ _ ^ ) V")

del X_text
gc.collect()

# Transform TF features to IDF features
# ============================================================================

tfidf_transformer = TfidfTransformer(sublinear_tf=True)

print("Transform to inverse document frequency (IDF) features...")
t0_transform = time()
X_tfidf = tfidf_transformer.fit_transform(X_tf)
print("Done in %.3f seconds" % (time() - t0_transform))
print("Transform finished! ( ^ _ ^ ) V")

# Extract topic model features
# ============================================================================

lda_params = {
    'n_components': num_classes,
    'learning_method': 'online',
    'n_jobs': -1,
    'random_state': 10
}
lda = LatentDirichletAllocation(**lda_params)
print("LDA vectorizer's hyper-parameters:")
pprint(lda_params)

print("Extract topic model features...")
t0_lda = time()
X_lda = lda.fit_transform(X_tf)
print("Done in %.3f seconds" % (time() - t0_lda))
print("Extract finished! ( ^ _ ^ ) V")

del X_tf
gc.collect()

# Concatenate TF-IDF features, embedding features and topic model features
# ============================================================================

print("Concatenate all the features...")
X = hstack([X_tfidf, csr_matrix(char_embed), csr_matrix(word_embed), csr_matrix(X_lda)], format='csr')

del X_tfidf, char_embed, word_embed, X_lda
gc.collect()

# Tuning the hyper-parameters of LightGBM model and save the results
# ============================================================================

print("Split data into training and validation set...")
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

df_params = pd.read_csv("lgb-tfidf-embedding-lda-params.csv")
num_params = df_params.shape[0]
print()
print("The number of parameter combinations is: %d" % num_params)

for i in range(num_params):
    print()
    print("Parameter combination %d" % (i + 1))

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
        'verbose': -1,
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

    res = "%s,%s,%d,%s,%.4f,%d,%d,%d,%.4f,%.4f,%d,%.4e,%.4e,%.4e,%.4e,%.4e,%d,%.5f,%.5f,%.2f,%.2f,%.5f,%.5f\n" % (
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "lgb-tfidf-embedding-%d" % (i + 1),  # model name
        tf_vect_params['max_features'] + 600 + num_classes,  # number of features
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

    f = open("lgb-tfidf-embedding-lda-tuning-results.csv", 'a')
    f.write(res)
    f.close()
