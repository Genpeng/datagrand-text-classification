# _*_ coding: utf-8 _*_

"""
Some utility functions for datagrand text process competition.

Author: StrongXGP (xgp1227@gmail.com)
Date:   2018/07/30
"""

import numpy as np


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


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Generate a batch iterator for a dataset."""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = ((len(data) - 1) // batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def to_categorical(y, num_classes=None):
    """Convert a class vector (integers) to binary class matrix.

    Parameters
    ----------
    y : class vector to be converted into a matrix
        (integers from 0 to `num_classes` - 1)
    num_classes : total number of classes

    Returns
    -------
    categorical : A binary matrix representation of `y`
    """
    y = np.array(y, dtype=np.int32)
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.int32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
