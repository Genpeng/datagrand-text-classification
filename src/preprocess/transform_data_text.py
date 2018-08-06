# _*_ coding: utf-8 _*_

"""
Transform the words (or characters) of all the samples to its ids.

Author: StrongXGP (xgp1227@gmail.com)
Date:   2018/07/30
"""

import pandas as pd
from time import time
from ..utils.data_utils import *


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
    print("Done in %.3f seconds." % (time() - t0_transform))
    print("[INFO] Finished!")

    # Save to file
    # ======================================================================================

    ids_train = pd.DataFrame(ids_train, dtype=np.int32)
    ids_train['class'] = pd.Series(labels_train, dtype=np.int32)
    ids_test = pd.DataFrame(ids_test, dtype=np.int32)

    print("[INFO] Save to file...")
    ids_train.to_csv("../../processed_data/train_ids_and_labels.txt", index=False)
    ids_test.to_csv("../../processed_data/test_ids.txt", index=False)
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
    ids_train = transform_to_ids(words_train, word_to_id_map)
    ids_test = transform_to_ids(words_test, word_to_id_map)
    print("[INFO] Finished!")

    # Save to file
    # ======================================================================================

    ids_train = pd.DataFrame(ids_train, dtype=np.int32)
    ids_train['class'] = pd.Series(labels_train, dtype=np.int32)
    ids_test = pd.DataFrame(ids_test, dtype=np.int32)

    print("[INFO] Save to file...")
    ids_train.to_csv("../../processed_data/train_ids_and_labels.txt", index=False)
    ids_test.to_csv("../../processed_data/test_ids.txt", index=False)
    print("[INFO] Finished! ( ^ _ ^ ) V")


if __name__ == '__main__':
    transform_characters()
    transform_words()
