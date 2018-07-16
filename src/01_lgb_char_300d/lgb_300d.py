# _*_ coding: utf-8 _*_

"""
Long text classification by using LightGBM model.

Author: StrongXGP
Date:	2018/07/13
"""

import gc
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Global variables
PAD_STR = '<PAD>'
SEQUENCE_LENGTH = 3000
EMBEDDING_SIZE = 300


def load_char_samples_and_labels(data_path, has_header=True, is_train=True):
    """Load characters of each sample (document)."""
    if has_header:
        start_index = 1
    else:
        start_index = 0

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()[start_index:]
        char_samples = [line.split(',')[1] for line in lines]
        char_samples = [char_sample.split() for char_sample in char_samples]

    if is_train:
        labels = [int(line.split(',')[3]) for line in lines]
    else:
        labels = []

    return char_samples, labels


def preprocess(data, sequence_length=3000):
    """Process the characters of each sample to a fixed length."""
    res = []
    for sample in data:
        if len(sample) > sequence_length:
            sample = sample[:sequence_length - 1]
            res.append(sample)
        else:
            str_added = [PAD_STR] * (sequence_length - len(sample))
            sample += str_added
            res.append(sample)
    return res


def generate_char_mapping(char_vectors_path):
    """Generate the mapping from characters to its corresponding vectors."""
    char_to_vec_map = {PAD_STR: np.zeros(EMBEDDING_SIZE, dtype=np.float32)}
    with open(char_vectors_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()[1:]
        lines = [line.split() for line in lines]
        for line in lines:
            word = line[0]
            if word not in char_to_vec_map:
                char_to_vec_map[word] = np.array(line[1:], dtype=np.float32)
    return char_to_vec_map


def generate_features(sample, char_to_vec_map):
    """Generate features by adding character vectors of each character in the sample."""
    np.random.seed(10)
    res = []
    for char in sample:
        if char in char_to_vec_map:
            res.append(char_to_vec_map[char])
        else:
            res.append(np.random.normal(size=(EMBEDDING_SIZE,)))
    matrix = np.concatenate(res).reshape([len(sample), -1])
    features = np.sum(matrix, axis=0)
    return features


def main():
    # Load data and process to a fixed length
    # ============================================================================

    print("Load data...")
    train_data_file = "../../raw_data/train_set.csv"
    test_data_file = "../../raw_data/test_set.csv"
    char_samples_train, labels_train = load_char_samples_and_labels(train_data_file, has_header=True, is_train=True)
    char_samples_test, _ = load_char_samples_and_labels(test_data_file, has_header=True, is_train=False)

    print("Process each sample to a fixed length...")
    char_samples_train = preprocess(char_samples_train, sequence_length=SEQUENCE_LENGTH)
    char_samples_test = preprocess(char_samples_test, sequence_length=SEQUENCE_LENGTH)

    # Generate the mapping from characters to its corresponding vectors
    # ============================================================================

    print("Generate the mapping from characters to its corresponding vectors...")
    char_vectors_path = "../../word_vectors/all/datagrand-char-300d.txt"
    char_to_vec_map = generate_char_mapping(char_vectors_path)

    # Extract features and split data into training, validation and testing set
    # ============================================================================

    print("Extract features...")
    num_train = len(char_samples_train)
    char_samples = char_samples_train + char_samples_test
    feature_vectors = []
    for char_sample in char_samples:
        feature_vector = generate_features(char_sample, char_to_vec_map)
        feature_vectors.append(feature_vector)

    print("Split data into training, validation and testing set...")
    feature_vectors_train = feature_vectors[:num_train]
    feature_vectors_test = feature_vectors[num_train:]

    X = pd.DataFrame(feature_vectors_train, dtype=np.float32)
    y = pd.Series(labels_train, dtype=np.int32) - 1
    indices_shuffled = np.random.permutation(np.arange(num_train))
    X_shuffled, y_shuffled = X.iloc[indices_shuffled], y.iloc[indices_shuffled]
    X_train, X_val, y_train, y_val = train_test_split(X_shuffled, y_shuffled, train_size=0.8, random_state=42)
    X_test = pd.DataFrame(feature_vectors_test, dtype=np.float32)

    del char_samples_train, char_samples_test, char_samples, char_to_vec_map
    del feature_vectors_train, feature_vectors_test, feature_vectors
    del X, y, X_shuffled, y_shuffled
    gc.collect()

    # Train LightGBM model
    # ============================================================================

    lgb_train = lgb.Dataset(X_train.values, y_train.values)
    lgb_val = lgb.Dataset(X_val.values, y_val.values, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': max(labels_train),
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'max_depth': 7,
        'learning_rate': 0.05
    }
    num_boost_round = 500
    feature_names = ['embed_' + str(col) for col in range(EMBEDDING_SIZE)]

    print("Start training...")
    start_time = time.time()
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=num_boost_round,
                    valid_sets=lgb_val,
                    feature_name=feature_names,
                    early_stopping_rounds=30)
    print("Training finished! ^_^")
    print("Total seconds: %ds" % (time.time() - start_time))

    # Calculate the f1 score of validation set
    probs_val = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    preds_val = np.argmax(probs_val, axis=1)
    score_val = f1_score(y_val, preds_val, average='weighted')
    print("The f1 score of validation set after %d epochs is: %f" % (gbm.best_iteration, score_val))

    print("Save model...")
    gbm.save_model("2018-07-15_lgb_300d.txt")

    # Make submission
    # ============================================================================

    df_test = pd.read_csv(test_data_file)
    submission = pd.DataFrame()
    submission['id'] = df_test['id']
    probs_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    preds_test = np.argmax(probs_test, axis=1) + 1
    submission['class'] = preds_test
    submission.to_csv("../../submissions/2018-07-15_lgb_submission.csv", index=False)


if __name__ == '__main__':
    main()
