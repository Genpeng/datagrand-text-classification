# _*_ coding: utf-8 _*_

"""
Baseline of datagrand text process competition.

Author: StrongXGP
Date:   2018/07/16
"""

import gc
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer


# Load data
# ============================================================================

train_data_file = "../raw_data/train_set.csv"
test_data_file = "../raw_data/test_set.csv"
df_train = pd.read_csv(train_data_file)
df_test = pd.read_csv(test_data_file)

X_train = df_train['word_seg']
y_train = df_train['class']
X_test = df_test['word_seg']
id_test = df_test['id']

del df_train, df_test
gc.collect()

# Extract features and train SVM model
# ============================================================================



column = "word_seg"
train = pd.read_csv('train_set.csv')
test = pd.read_csv('test_set.csv')
test_id = test["id"].copy()
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])
fid0 = open('baseline.csv', 'w')

y = (train["class"] - 1).astype(int)
lin_clf = svm.LinearSVC()
lin_clf.fit(trn_term_doc, y)
preds = lin_clf.predict(test_term_doc)
i = 0
fid0.write("id,class" + "\n")
for item in preds:
    fid0.write(str(i) + "," + str(item + 1) + "\n")
    i = i + 1
fid0.close()
