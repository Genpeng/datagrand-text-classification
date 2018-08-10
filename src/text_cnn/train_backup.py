# _*_ coding: utf-8 _*_

"""
The training procedure of TextCNN model.

Author: StrongXGP (xgp1227#163.com)
Date:   2018/07/31
"""

import gc
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from time import time
from datetime import datetime

# Import self-defined modules and classes
from util import *
from text_cnn import TextCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================================================
# Parameters

# Data loading parameters
tf.flags.DEFINE_float("val_sample_percentage", 0.1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_data_file", "../../processed_data/word/train_ids_and_labels.txt", "File of training data set")
tf.flags.DEFINE_string("embedding_file", "../../embeddings/word-embedding-300d-mc5.npy", "File of embedding lookup table")

# Model hyper-parameters
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: 2,3,4)")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 256)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate (default: 0.001)")
tf.flags.DEFINE_integer("batch_size", 128, "Batch size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate the model on the validation set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 1000)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 2)")
# tf.flags.DEFINE_integer("update_embed_every", 1, "Start update embedding loopup table after this many epochs (default: 1)")
# tf.flags.DEFINE_float("decay_rate", 0.8, "Decay rate of the learning rate (default: 0.8)")
# tf.flags.DEFINE_integer("decay_every", 15000, "Decay the learning rate after this many steps (default: 15000)")

# Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# =============================================================================================


def load_data_and_embedding():
    """Prepare data and embedding lookup table for training procedure."""

    # Load data
    df_data = pd.read_csv(FLAGS.train_data_file)
    y = df_data['class'] - 1  # class (0 ~ 18)
    X = df_data.drop(['class'], axis=1).values

    # Transform to binary class matrix
    y = to_categorical(y.values)

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(range(len(y)))
    X_shuffled = X[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split to train/test set
    # TODO: This is very crude, should use cross validation
    val_sample_index = -1 * int(FLAGS.val_sample_percentage * len(y))
    X_train, X_val = X_shuffled[:val_sample_index], X_shuffled[val_sample_index:]
    y_train, y_val = y_shuffled[:val_sample_index], y_shuffled[val_sample_index:]

    del df_data, X, y, X_shuffled, y_shuffled
    gc.collect()

    # Load embedding lookup table
    embedding_lookup_table = np.load(FLAGS.embedding_file)
    return X_train, y_train, X_val, y_val, embedding_lookup_table


def train(X_train, y_train, X_val, y_val, embedding_lookup_table):
    """Training procedure."""
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(sequence_length=X_train.shape[1],
                          num_classes=y_train.shape[1],
                          embedding_lookup_table=embedding_lookup_table,
                          filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                          num_filters=FLAGS.num_filters,
                          l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define training procedure
            global_step = tf.Variable(0, trainable=False, name="global_step")
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = list()
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram(name="{}/grad/hist".format(v.name), values=g)
                    sparsity_summary = tf.summary.scalar(name="{}/grad/sparsity".format(v.name),
                                                         tensor=tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar(name="loss", tensor=cnn.loss)
            acc_summary = tf.summary.scalar(name="accuracy", tensor=cnn.accuracy)

            # Output directories for models and summaries
            timestamp = str(int(time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Write to {}\n".format(out_dir))

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validate summaries
            val_summary_op = tf.summary.merge([loss_summary, acc_summary])
            val_summary_dir = os.path.join(out_dir, "summaries", "val")
            val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

            # Checkpoint directory. TensorFlow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(X_batch, y_batch):
                """A single training step."""
                feed_dict = {cnn.input_X: X_batch,
                             cnn.input_y: y_batch,
                             cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict=feed_dict)
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("{} - step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                # print("{} - step {:>6d}, loss {:8.5f}, acc {:.2%}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, global_step=step)

            def val_step(X_batch, y_batch, writer=None):
                """Evaluate model on the validation set."""
                feed_dict = {cnn.input_X: X_batch,
                             cnn.input_y: y_batch,
                             cnn.dropout_keep_prob: 1.0}
                step, summaries, loss, accuracy = sess.run([global_step, val_summary_op, cnn.loss, cnn.accuracy],
                                                           feed_dict=feed_dict)
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("{} - step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                # print("{} - step {:>6d}, loss {:8.5f}, acc {:.2%}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, global_step=step)

            # Generate batches
            batches = batch_iter(list(zip(X_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop
            for batch in batches:
                X_batch, y_batch = zip(*batch)
                train_step(X_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation: ")
                    val_step(X_val, y_val, writer=val_summary_writer)
                    print()
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    print("Load data and embedding lookup table...")
    X_train, y_train, X_val, y_val, embedding_lookup_table = load_data_and_embedding()
    print("Load finished!\n")

    print("Start training...")
    t0 = time()
    train(X_train, y_train, X_val, y_val, embedding_lookup_table)
    print("Done in %.3f seconds." % (time() - t0))
    print("Training finished! ( ^ _ ^ ) V")


if __name__ == '__main__':
    tf.app.run()
