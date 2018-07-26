# _*_ coding: utf-8 _*_

"""
Train a SVM model with TF-IDF feature vectors for datagrand text process competition.

Author: StrongXGP
Date:   2018/07/20
"""

import gc
import pandas as pd
from time import time
from pprint import pprint
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================================
# Load data

print("Loading data...")
train_data_file = "../../raw_data/train_demo.csv"
test_data_file = "../../raw_data/test_demo.csv"
df_train = pd.read_csv(train_data_file)
df_test = pd.read_csv(test_data_file)

text_train = df_train['word_seg']  # words of training samples (documents)
y_train = df_train['class']  # labels of training samples (1 ~ 19)
text_test = df_test['word_seg']  # words of testing samples (documents)
id_test = df_test['id']
num_classes = max(y_train)
print("The number of training set is: %d" % len(text_train))
print("The number of testing set is: %d" % len(text_test))
print("The number of classes is: %d" % num_classes)

del df_train, df_test
gc.collect()

# ============================================================================
# Extract features and train SVM model

# Initialize feature extractor
vect_params = {
    'ngram_range': (1, 2),
    'min_df': 5,
    'max_df': 0.9,
    'max_features': 200000,
    'sublinear_tf': True
}
print("TfidfVectorizer's hyper-parameters:")
pprint(vect_params)
vectorizer = TfidfVectorizer(**vect_params)

print("Extract features...")
t0_extract = time()
X_train = vectorizer.fit_transform(text_train)
X_test = vectorizer.transform(text_test)
print("Done in %.3f seconds" % (time() - t0_extract))
print("Extract finished! ( ^ _ ^ ) V")

print("Start training...")
clf = LinearSVC()
t0_train = time()
clf.fit(X_train, y_train - 1)
print("Done in %.3f seconds" % (time() - t0_train))
print("Training finish! ( ^ _ ^ ) V ")

pred_train = clf.predict(X_train) + 1
acc_train = accuracy_score(y_train, pred_train)
f1_train = f1_score(y_train, pred_train, average='weighted')
print("Train Accuracy: %.2f, Train F1 Score: %.5f" % (acc_train * 100, f1_train))

# ============================================================================
# Make submission

pred_test = clf.predict(X_test) + 1
submission = pd.DataFrame()
submission['id'] = id_test
submission['class'] = pred_test
submission.to_csv("2018-07-20_svm-tfidf-submission.csv", index=False)
