# _*_ coding: utf-8 _*_

"""
Generate features of training and testing set.

Author: StrongXGP
Date:	2018/07/16
"""

import numpy as np
import pandas as pd

# Global variables
PAD_STR = '<pad>'
SEQUENCE_LENGTH = 2000  # documents with the number of words less than 2000 is 95.3147%
EMBEDDING_SIZE = 300


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
            sample = sample[:sequence_length - 1]
            res.append(sample)
        else:
            str_added = [PAD_STR] * (sequence_length - len(sample))
            sample += str_added
            res.append(sample)
    return res


def generate_word_mapping(char_vectors_path):
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


def generate_features_add(sample, word_to_vec_map):
    """Generate features by adding word vectors of each word in the sample."""
    np.random.seed(10)
    res = []
    for word in sample:
        if word in word_to_vec_map:
            res.append(word_to_vec_map[word])
        else:
            res.append(np.random.normal(size=(EMBEDDING_SIZE,)))
    matrix = np.concatenate(res).reshape([len(sample), -1])
    features = np.sum(matrix, axis=0)
    return features


def generate_features_mean(sample, word_to_vec_map):
    """Generate features by averaging word vectors of each word in the sample."""
    np.random.seed(10)
    res = []
    for word in sample:
        if word in word_to_vec_map:
            res.append(word_to_vec_map[word])
        else:
            res.append(np.random.normal(size=(EMBEDDING_SIZE,)))
    matrix = np.concatenate(res).reshape([len(sample), -1])
    features = np.mean(matrix, axis=0)
    return features


def plan1():
    """
    Plan 1
    ------

    Each sentence is processed to a fixed length (here is 3000), if the length of sentence
    is greater than 3000, it is truncated to 3000; if the sentence is not long enough, it
    is complemented with a word ('<pad>'). After that, summing all the word
    vectors in a document as its feature vector.
    """

    # Load data and process to a fixed length
    print("Load data...")
    train_data_file = "../../raw_data/train_set.csv"
    test_data_file = "../../raw_data/test_set.csv"
    word_samples_train, labels_train = load_word_samples_and_labels(train_data_file, header=True, train=True)
    word_samples_test, _ = load_word_samples_and_labels(test_data_file, header=True, train=False)

    print("Process each sample to a fixed length...")
    word_samples_train = preprocess(word_samples_train, sequence_length=SEQUENCE_LENGTH)
    word_samples_test = preprocess(word_samples_test, sequence_length=SEQUENCE_LENGTH)

    # Generate the mapping from words to its corresponding vectors
    print("Generate the mapping from words to its corresponding vectors...")
    word_vectors_path = "../../word_vectors/all/datagrand-word-300d-mc5.txt"
    word_to_vec_map = generate_word_mapping(word_vectors_path)

    # Extract features
    print("Extract features...")
    num_train = len(word_samples_train)
    word_samples = word_samples_train + word_samples_test
    feature_vectors = []
    for word_sample in word_samples:
        feature_vector = generate_features_add(word_sample, word_to_vec_map)
        feature_vectors.append(feature_vector)

    feature_vectors_train = feature_vectors[:num_train]
    feature_vectors_test = feature_vectors[num_train:]

    X_train = pd.DataFrame(feature_vectors_train, dtype=np.float32)
    X_train['class'] = pd.Series(labels_train, dtype=np.int32)
    X_test = pd.DataFrame(feature_vectors_test, dtype=np.float32)

    # Save to CSV file
    print("Saving to csv file...")
    X_train.to_csv("../../processed_data/train-data-word-300d-sum.txt", index=False)
    X_test.to_csv("../../processed_data/test-data-word-300d-sum.txt", index=False)


def plan2():
    """
    Plan 2
    ------

    Despite the length of the document, averaging all the word vectors
    in a document as its feature vector.
    """

    # Load data
    print("Load data...")
    train_data_file = "../../raw_data/train_set.csv"
    test_data_file = "../../raw_data/test_set.csv"
    word_samples_train, labels_train = load_word_samples_and_labels(train_data_file, header=True, train=True)
    word_samples_test, _ = load_word_samples_and_labels(test_data_file, header=True, train=False)

    # Generate the mapping from words to its corresponding vectors
    print("Generate the mapping from words to its corresponding vectors...")
    word_vectors_path = "../../word_vectors/all/datagrand-word-300d-mc5.txt"
    word_to_vec_map = generate_word_mapping(word_vectors_path)

    # Extract features
    print("Extract features...")
    num_train = len(word_samples_train)
    word_samples = word_samples_train + word_samples_test
    feature_vectors = []
    for word_sample in word_samples:
        feature_vector = generate_features_mean(word_sample, word_to_vec_map)
        feature_vectors.append(feature_vector)

    feature_vectors_train = feature_vectors[:num_train]
    feature_vectors_test = feature_vectors[num_train:]

    X_train = pd.DataFrame(feature_vectors_train, dtype=np.float32)
    X_train['class'] = pd.Series(labels_train, dtype=np.int32)
    X_test = pd.DataFrame(feature_vectors_test, dtype=np.float32)

    # Save to CSV file
    print("Saving to csv file...")
    X_train.to_csv("../../processed_data/train-data-word-300d-mean.txt", index=False)
    X_test.to_csv("../../processed_data/test-data-word-300d-mean.txt", index=False)


if __name__ == '__main__':
    plan1()
    plan2()
