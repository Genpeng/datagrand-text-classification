# _*_ coding: utf-8 _*_

import numpy as np


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
    y = np.array(y, dtype='int')
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