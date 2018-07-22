# _*_ coding: utf-8 _*_

"""
Baseline (0.77799) of datagrand text process competition.

reference:
1. http://www.dcjingsai.com/common/bbs/topicDetails.html?tid=1522

Author: StrongXGP
Date:   2018/07/16
"""

import gc
import time
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================================
# Load data

print("Loading data...")
train_data_file = "../../raw_data/train_set.csv"
test_data_file = "../../raw_data/test_set.csv"
df_train = pd.read_csv(train_data_file)
df_test = pd.read_csv(test_data_file)

X_train = df_train['word_seg']
y_train = df_train['class']
X_test = df_test['word_seg']
id_test = df_test['id']

del df_train, df_test
gc.collect()

# ============================================================================
# Extract features and train SVM model

print("Extract features...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
bigram_tfidf_train = vectorizer.fit_transform(X_train)
bigram_tfidf_test = vectorizer.transform(X_test)

print("Start training...")
clf = LinearSVC()
t0 = time.time()
clf.fit(bigram_tfidf_train, y_train - 1)
print("Total second: %ds" % (time.time() - t0))
print("Training finish! ( ^ _ ^ ) V ")

# ============================================================================
# Make submission

y_test = clf.predict(bigram_tfidf_test) + 1
submission = pd.DataFrame()
submission['id'] = id_test
submission['class'] = y_test
submission.to_csv("submission.csv", index=False)
