# _*_ coding: utf-8 _*_

"""
Search the best parameters for `TfidfVectorizer` and `LinearSVC`.

Author: StrongXGP (xgp1227#gmail.com)
Date:   2018/07/18
"""

import gc
import pandas as pd
from time import time
from pprint import pprint
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# ============================================================================
# Load data

print("Load data...")
data = pd.read_csv("../../raw_data/train_set.csv")

X_train = data['word_seg']
y_train = data['class']

del data
gc.collect()

# ============================================================================
# Define a pipeline combining a text feature extractor with a SVM classifier

pipeline = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
    ('clf', LinearSVC(random_state=42))
])

parameters = {
    'vect__max_df': (0.8, 0.9),
    'vect__min_df': (3, 5, 10, 20),
    'vect__max_features': (50000, 100000, 150000, None),
    # 'vect__ngram_range': ((1, 1), (1, 2)),
    'tfidf__sublinear_tf': (True, False),
    # 'clf__penalty': ('l2', 'l1'),
    'clf__loss': ('hinge', 'squared_hinge'),
    'clf__max_iter': (500, 1000)
}

if __name__ == '__main__':
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("Done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
