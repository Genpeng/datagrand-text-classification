# _*_ coding: utf-8 _*_

"""
Transform the words (or characters) of all the samples to its ids.

Author: StrongXGP (xgp1227@gmail.com)
Date:   2018/07/30
"""

import pickle
import numpy as np
import pandas as pd

# Global variable
PAD_STR = '<PAD>'
SEQUENCE_LENGTH = 2000  # documents with the number of words less than 2000 is 95.3147%


def load_word_samples_and_labels(data_path, header=True, train=True):
    """Load words and labels of each sample (document)."""
    if header:
        start_index = 1
    else:
        start_index = 0

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()[start_index:]
        word_samples = [line.split(',')[2] for line in lines]
        word_samples = [word_sample.split() for word_sample in word_samples]

    if train:
        labels = [int(line.split(',')[3]) for line in lines]
    else:
        labels = []

    return word_samples, labels


def preprocess(data, sequence_length=3000):
    """Process the words of each sample to a fixed length."""
    res = []
    for sample in data:
        if len(sample) > sequence_length:
            sample = sample[:sequence_length]
            res.append(sample)
        else:
            str_added = [PAD_STR] * (sequence_length - len(sample))
            sample += str_added
            res.append(sample)
    return res


def transform_to_ids(data, word_to_id_map):
    """Transform the words (characters) of a sample to its ids."""
    res = list()
    for words in data:
        ids = list()
        for word in words:
            if word in word_to_id_map:
                ids.append(word_to_id_map[word])
            else:
                ids.append(1)  # 1 is the id of '<UNK>'
        res.append(ids)
    return res


# Load the mapping from words to its corresponding ids
# ======================================================================================

print("Load the mapping from words to its corresponding ids...")
word2id_file = "../../processed_data/word2id.pkl"
with open(word2id_file, 'rb') as fin:
    word_to_id_map = pickle.load(fin)

# Load data, truncate to fixed length and transform to ids
# ======================================================================================

print("Load data...")
train_data_file = "../../raw_data/train_set.csv"
test_data_file = "../../raw_data/test_set.csv"
words_train, labels_train = load_word_samples_and_labels(train_data_file, header=True, train=True)
words_test, _ = load_word_samples_and_labels(test_data_file, header=True, train=False)

print("Truncate to fixed length...")
words_train = preprocess(words_train, sequence_length=SEQUENCE_LENGTH)
words_test = preprocess(words_test, sequence_length=SEQUENCE_LENGTH)

print("Transform to ids...")
ids_train = transform_to_ids(words_train, word_to_id_map)
ids_test = transform_to_ids(words_test, word_to_id_map)

# Save to file
# ======================================================================================

ids_train = pd.DataFrame(ids_train, dtype=np.int32)
ids_train['class'] = pd.Series(labels_train, dtype=np.int32)
ids_test = pd.DataFrame(ids_test, dtype=np.int32)

print("Save to file...")
ids_train.to_csv("../../processed_data/train_ids_and_labels.txt", index=False)
ids_test.to_csv("../../processed_data/test_ids.txt", index=False)
print("Finished! ( ^ _ ^ ) V")
