# _*_ coding: utf-8 _*_

"""
Train LightGBM model with TF-IDF features and embedding features as feature vectors.

Author: StrongXGP
Date:	2018/07/22
"""

import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from time import time
from pprint import pprint
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
# ============================================================================

print("Load data...")
data_file = "../../raw_data/train_set.csv"
data = pd.read_csv(data_file)

X_text = data['word_seg']
y_train = data['class'] - 1
num_classes = max(y_train) + 1
print("The number of samples is: %d" % len(X_text))
print("The number of classes is: %d" % num_classes)

del data
gc.collect()

# Load character and word embedding
# ============================================================================

print("Load character and word embedding...")
char_embed_file = "../../processed_data/train-data-char-300d-mean.txt"
char_embed = pd.read_csv(char_embed_file).drop(['class'], axis=1)
word_embed_file = "../../processed_data/train-data-word-300d-mean.txt"
word_embed = pd.read_csv(word_embed_file).drop(['class'], axis=1)

# Extract TF-IDF features
# ============================================================================

vect_params = {
    'ngram_range': (1, 2),
    'min_df': 5,
    'max_df': 0.9,
    'max_features': 200000,
    'sublinear_tf': True
}
vectorizer = TfidfVectorizer(**vect_params)
print("Vectorizer's hyper-parameters:")
pprint(vect_params)

print("Extract features...")
t0_extract = time()
X_train = vectorizer.fit_transform(X_text)
print("Done in %.3f seconds" % (time() - t0_extract))
print("Extract finished! ( ^ _ ^ ) V")

del X_text
gc.collect()

# Concatenate TF-IDF features and embedding features
# ============================================================================

print("Concatenate TF-IDF features and embedding features...")
X_train = hstack([X_train, csr_matrix(char_embed), csr_matrix(word_embed)], format='csr')

del char_embed, word_embed
gc.collect()

# Train the LightGBM model
# ============================================================================

lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=True)

df_params = pd.read_csv("lgb-tfidf-embedding-tuning-results.csv").sort_values(by='f1_val', ascending=False)
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
        'verbose': -1,
        'is_training_metric': 'True'
    }
    print("LightGBM's hyper-parameters:")
    pprint(params)

    num_epochs = df_params['best_iter'].values[i]
    print("Round number: %d" % num_epochs)

    print("Start training...")
    t0_train = time()
    evals_result = {}
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=num_epochs,
                    valid_sets=lgb_train,
                    valid_names=['train'],
                    evals_result=evals_result,
                    verbose_eval=100)
    print("Done in %.3f seconds" % (time() - t0_train))
    print("Training finished! ( ^ _ ^ ) V")

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
    feature_importance.to_csv("lgb-tfidf-embedding-feature-importance.csv", index=False)

    # Save model
    gbm.save_model("2018-07-22_lgb-tfidf-embedding-model.txt", num_iteration=best_iter)
