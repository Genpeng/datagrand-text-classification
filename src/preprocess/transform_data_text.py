# _*_ coding: utf-8 _*_

"""
Transform the words (or characters) of all the samples to its ids.

Author: StrongXGP (xgp1227@gmail.com)
Date:   2018/07/30
"""

import os
import pickle
import numpy as np
import pandas as pd
from time import time

# Global variable
PAD_STR = '<PAD>'


def load_mapping_table(mapping_table_file):
    """Load the mapping table from words (characters) to its corresponding ids."""
    with open(mapping_table_file, 'rb') as fin:
        mapping_table = pickle.load(fin)
    return mapping_table


def load_samples_and_labels(data_path, header=True, col=1, train=True):
    """Load words (or characters) and its label of all the samples."""
    if header:
        start_index = 1
    else:
        start_index = 0

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()[start_index:]
        samples = [line.split(',')[col] for line in lines]
        samples = [sample.split() for sample in samples]

    if train:
        labels = [int(line.split(',')[3]) for line in lines]
    else:
        labels = []

    return samples, labels


def truncate(data, sequence_length=3000):
    """Truncate the words (characters) of each sample to a fixed length."""
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
            ids.append(word_to_id_map.get(word, 1))  # 1 is the id of '<UNK>'
            # if word in word_to_id_map:
            #     ids.append(word_to_id_map[word])
            # else:
            #     ids.append(1)  # 1 is the id of '<UNK>'
        res.append(ids)
    return res


def transform_characters():
    # Load the mapping table from characters to its ids
    # ======================================================================================

    print("[INFO] Load the mapping from characters to its corresponding ids...")
    char_to_id_file = "../../embeddings/char2id.pkl"
    char_to_id_map = load_mapping_table(char_to_id_file)
    print("[INFO] Finished!")

    # Load data, truncate to fixed length and transform to ids
    # ======================================================================================

    print("[INFO] Loading data...")
    train_data_file = "../../raw_data/train_set.csv"
    test_data_file = "../../raw_data/test_set.csv"
    chars_train, labels_train = load_samples_and_labels(train_data_file, header=True, col=1, train=True)
    chars_test, _ = load_samples_and_labels(test_data_file, header=True, col=1, train=False)
    print("[INFO] Finished!")

    print("[INFO] Truncate to fixed length...")
    char_sequence_length = 3000
    chars_train = truncate(chars_train, sequence_length=char_sequence_length)
    chars_test = truncate(chars_test, sequence_length=char_sequence_length)
    print("[INFO] Finished!")

    print("[INFO] Transform characters to its corresponding ids...")
    t0_transform = time()
    ids_train = transform_to_ids(chars_train, char_to_id_map)
    ids_test = transform_to_ids(chars_test, char_to_id_map)
    print("[INFO] Done in %.3f seconds." % (time() - t0_transform))
    print("[INFO] Finished!")

    # Save to file
    # ======================================================================================

    ids_train = pd.DataFrame(ids_train, dtype=np.int32)
    ids_train['class'] = pd.Series(labels_train, dtype=np.int32)
    ids_test = pd.DataFrame(ids_test, dtype=np.int32)

    print("[INFO] Save to file...")
    ids_train.to_csv("../../processed_data/char/train_ids_and_labels.txt", index=False)
    ids_test.to_csv("../../processed_data/char/test_ids.txt", index=False)
    print("[INFO] Finished! ( ^ _ ^ ) V")


def transform_words():
    # Load the mapping table from words to its ids
    # ======================================================================================

    print("[INFO] Load the mapping from words to its corresponding ids...")
    word_to_id_file = "../../embeddings/word2id.pkl"
    word_to_id_map = load_mapping_table(word_to_id_file)
    print("[INFO] Finished!")

    # Load data, truncate to fixed length and transform to ids
    # ======================================================================================

    print("[INFO] Load data...")
    train_data_file = "../../raw_data/train_set.csv"
    test_data_file = "../../raw_data/test_set.csv"
    words_train, labels_train = load_samples_and_labels(train_data_file, header=True, col=2, train=True)
    words_test, _ = load_samples_and_labels(test_data_file, header=True, col=2, train=False)
    print("[INFO] Finished!")

    print("[INFO] Truncate to fixed length...")
    word_sequence_length = 2000  # documents with the number of words less than 2000 is 95.3147%
    words_train = truncate(words_train, sequence_length=word_sequence_length)
    words_test = truncate(words_test, sequence_length=word_sequence_length)
    print("[INFO] Finished!")

    print("[INFO] Transform to ids...")
    t0_transform = time()
    ids_train = transform_to_ids(words_train, word_to_id_map)
    ids_test = transform_to_ids(words_test, word_to_id_map)
    print("[INFO] Done in %.3f seconds." % (time() - t0_transform))
    print("[INFO] Finished!")

    # Save to file
    # ======================================================================================

    ids_train = pd.DataFrame(ids_train, dtype=np.int32)
    ids_train['class'] = pd.Series(labels_train, dtype=np.int32)
    ids_test = pd.DataFrame(ids_test, dtype=np.int32)

    print("[INFO] Save to file...")
    ids_train.to_csv("../../processed_data/word/train_ids_and_labels.txt", index=False)
    ids_test.to_csv("../../processed_data/word/test_ids.txt", index=False)
    print("[INFO] Finished! ( ^ _ ^ ) V")


if __name__ == '__main__':
    transform_characters()
    transform_words()
