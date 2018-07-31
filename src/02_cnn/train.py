# _*_ coding: utf-8 _*_

"""
The training procedure of TextCNN model.

Author: StrongXGP (xgp1227#163.com)
Date:   2018/07/31
"""

import gc
import os
import pandas as pd
import tensorflow as tf
from time import time
from datetime import datetime

# Import self-defined modules and classes
from text_cnn import TextCNN
from ..util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
# =============================================================================================

# Data loading parameters
tf.flags.DEFINE_float("val_sample_percentage", 0.1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_data_file", "../../processed_data/train_ids_and_labels.txt", "File of training data set")
tf.flags.DEFINE_string("embedding_file", "../../word_vectors/word_vectors/word-embedding-300d-mc5.npy", "File of embedding lookup table")

# Model hyper-parameters
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: 2,3,4)")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 256)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate (default: 0.001)")
tf.flags.DEFINE_float("decay_rate", 0.8, "Decay rate of the learning rate (default: 0.8)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate the model on the validation set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 1000)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 3)")
tf.flags.DEFINE_integer("update_embed_every", 1, "Start update embedding loopup table after this many epochs (default: 1)")
tf.flags.DEFINE_integer("decay_every", 15000, "Decay the learning rate after this many steps (default: 15000)")

# Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def preprocess():
    # Data preparation
    # =========================================================================================

    print("Load data...")
    df_data = pd.read_csv(FLAGS.train_data_file)
    y = df_data['class'] - 1  # label (0 ~ 18)
    X = df_data.drop(['class'], axis=1)

    #


if __name__ == '__main__':
    preprocess()
