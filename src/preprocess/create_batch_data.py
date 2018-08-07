# _*_ coding: utf-8 _*_

"""
Split training training set and testing set into small batches.

Author: StrongXGP (xgp1227@gmail.com)
Date:   2018/08/07
"""

import os
import numpy as np
import pandas as pd


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


def save_train_data_to_batches(save_dir, X, y, batch_size=128):
    """Save training (or validation) set to small batches."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_samples = len(X)
    num_batches = ((num_samples - 1) // batch_size) + 1
    for i in range(num_batches):
        start_index = batch_size * i
        end_index = min(batch_size * (i + 1), num_samples)
        save_path = os.path.join(save_dir, '%d.npz' % i)
        X_batch, y_batch = X[start_index:end_index], y[start_index:end_index]
        np.savez(save_path, X=X_batch, y=y_batch)


def save_test_data_to_batches(save_dir, X, batch_size=128):
    """Save testing set to small batches."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_samples = len(X)
    num_batches = ((num_samples - 1) // batch_size) + 1
    for i in range(num_batches):
        start_index = batch_size * i
        end_index = min(batch_size * (i + 1), num_samples)
        save_path = os.path.join(save_dir, '%d.npz' % i)
        X_batch = X[start_index:end_index]
        np.savez(save_path, X=X_batch)


def generate_batches(data_dir):
    # Prepare data
    # ========================================================================

    print("[INFO] Prepare data...")

    train_data_file = os.path.join(data_dir, "train_ids_and_labels.txt")
    test_data_file = os.path.join(data_dir, "test_ids.txt")
    data_train = pd.read_csv(train_data_file)
    X_test = pd.read_csv(test_data_file).values

    y = data_train['class'].values - 1
    X = data_train.drop(['class'], axis=1).values

    # Convert a class vector (integers) to binary class matrix
    y = to_categorical(y)

    # Randomly shuffle data
    np.random.seed(42)
    indices_shuffled = np.random.permutation(range(len(y)))
    X_shuffled = X[indices_shuffled]
    y_shuffled = y[indices_shuffled]

    # Split training data into training set and validation set
    val_sample_percentage = 0.1
    val_sample_index = -1 * int(val_sample_percentage * len(y))
    X_train, X_val = X_shuffled[:val_sample_index], X_shuffled[val_sample_index:]
    y_train, y_val = y_shuffled[:val_sample_index], y_shuffled[val_sample_index:]

    print("[INFO] Finished!")

    # Save to file
    # ========================================================================

    print("[INFO] Save to file...")

    # Save training set to small batches
    save_dir_train = os.path.join(data_dir, 'train')
    if not os.path.exists(save_dir_train):
        os.makedirs(save_dir_train)
    save_train_data_to_batches(save_dir_train, X_train, y_train, batch_size=128)

    # Save validation set to small batches
    save_dir_val = os.path.join(data_dir, 'val')
    if not os.path.exists(save_dir_val):
        os.makedirs(save_dir_val)
    save_train_data_to_batches(save_dir_val, X_val, y_val, batch_size=128)

    # Save testing set to small batches
    save_dir_test = os.path.join(data_dir, 'test')
    if not os.path.exists(save_dir_test):
        os.makedirs(save_dir_test)
    save_test_data_to_batches(save_dir_test, X_test, batch_size=128)

    print("[INFO] Finished!")


def main():
    char_data_dir = "../../processed_data/char/"
    generate_batches(char_data_dir)

    word_data_dir = "../../processed_data/word/"
    generate_batches(word_data_dir)


if __name__ == '__main__':
    main()
