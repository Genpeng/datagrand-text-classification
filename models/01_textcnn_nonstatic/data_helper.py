# _*_ coding: utf-8 _*_

"""
An utility module for preprocessing data.

Author: StrongXGP
Date:   2018/07/05
"""

import numpy as np


def load_data_and_labels(data_file):
    """Load the dataset of datagrand text classification competition."""
    lines = open(data_file, 'r', encoding='utf-8').read().splitlines()[1:]
    x_text = [line.split(',')[1] for line in lines]
    labels = [int(line.split(',')[3]) for line in lines]
    # Generate labels
    num_classes = max(labels)
    y = np.zeros((len(labels), num_classes), dtype=np.int32)
    for n, label in enumerate(labels):
        y[n, label-1] = 1
    return x_text, y


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Generates a batch iterator for a dataset."""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
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
