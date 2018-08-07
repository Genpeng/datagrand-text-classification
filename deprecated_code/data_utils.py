# _*_ coding: utf-8 _*_

"""
Utilities for data loading and pre-processing.

Author: StrongXGP (xgp1227@gmail.com)
Date:   2018/08/06
"""

import numpy as np

# Global variable
PAD_STR = '<PAD>'


class DataUtil:
    """A assistant class related to data loading and pre-processing."""
    def __init__(self):
        self._char_to_id_map = None
        self._word_to_id_map = None

    @property
    def char_to_id_map(self):
        return self._char_to_id_map

    @char_to_id_map.setter
    def char_to_id_map(self, char_to_id_map):
        self._char_to_id_map = char_to_id_map

    @property
    def word_to_id_map(self):
        return self._word_to_id_map

    @word_to_id_map.setter
    def word_to_id_map(self, word_to_id_map):
        self._word_to_id_map = word_to_id_map

    @staticmethod
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

    @staticmethod
    def batch_iter(data, batch_size, num_epochs, shuffle=True):
        """Generate a batch iterator for a dataset."""
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
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

    @staticmethod
    def to_categorical(y, num_classes=None):
        """Convert a class vector (integers) to binary class matrix.

        Args:
            y: class vector to be converted into a matrix
            (integers from 0 to `num_classes` - 1)
            num_classes: total number of classes

        Returns:
            categorical: A binary matrix representation of `y`
        """
        y = np.array(y, dtype=np.int32)
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=np.float32)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    @staticmethod
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

    # @classmethod
    # def transform_words_to_ids(cls, data):
    #     """Transform the words of all the samples to its corresponding ids."""
    #
    #
    #
    #     return map(DataUtil._transform_one_words_sample, data)
    #     # res = list()
    #     # for words in data:
    #     #     ids = map()
    #     #     ids = list()
    #     #     for word in words:
    #     #         ids.append(word_to_id_map.get(word, 1))  # 1 is the id of '<UNK>'
    #     #         # if word in word_to_id_map:
    #     #         #     ids.append(word_to_id_map[word])
    #     #         # else:
    #     #         #     ids.append(1)  # 1 is the id of '<UNK>'
    #     #     res.append(ids)
    #     # return res
    #
    # @staticmethod
    # def _transform_one_words_sample(sample):
    #     """Transform the words of a sample to its ids."""
    #     ids = list()
    #     for word in sample:
    #         ids.append()
