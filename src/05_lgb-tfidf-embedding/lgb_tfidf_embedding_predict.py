# _*_ coding: utf-8 _*_

"""
Load LightGBM model, predict the test set and make submission.

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
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
# ============================================================================

print("Load data...")
test_data_file = "../../raw_data/test_set.csv"
df_test = pd.read_csv(test_data_file)
text_test = df_test['word_seg']
id_test = df_test['id']

del df_test
gc.collect()

# Load character and word embedding
# ============================================================================

print("Load character and word embedding...")
char_embed_test_file = "../../processed_data/test-data-char-300d-mean.txt"
char_embed_test = pd.read_csv(char_embed_test_file)
word_embed_test_file = "../../processed_data/test-data-word-300d-mean.txt"
word_embed_test = pd.read_csv(word_embed_test_file)

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
X_test = vectorizer.fit_transform(text_test)
print("Done in %.3f seconds" % (time() - t0_extract))
print("Extract finished! ( ^ _ ^ ) V")

del text_test
gc.collect()

# Concatenate TF-IDF features and embedding features
# ============================================================================

print("Concatenate TF-IDF features and embedding features...")
X_test = hstack([X_test, csr_matrix(char_embed_test), csr_matrix(word_embed_test)], format='csr')

del char_embed_test, word_embed_test
gc.collect()

# Load LightGBM model, predict test set and make submission
# ============================================================================

gbm = lgb.Booster(model_file="2018-07-22_lgb-tfidf-embedding-model.txt")

probs_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)
preds_test = np.argmax(probs_test, axis=1) + 1

submission = pd.DataFrame()
submission['id'] = id_test
submission['class'] = preds_test
submission.to_csv("2018-07-23_lgb-tfidf-embedding-submission.csv", index=False)
