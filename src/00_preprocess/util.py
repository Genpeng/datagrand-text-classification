# _*_ coding: utf-8 _*_

"""
Some utility functions for datagrand text process competition.

Author: StrongXGP (xgp1227@gmail.com)
Date:   2018/07/30
"""


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
