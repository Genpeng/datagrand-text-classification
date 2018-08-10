# _*_ coding: utf-8 _*_

"""
The training procedure of TextCNN model.

Author: StrongXGP (xgp1227#163.com)
Date:   2018/07/31
"""

import os
import numpy as np
import tensorflow as tf
from time import time
from datetime import datetime

# Import self-defined modules and classes
from util import *
from text_cnn import TextCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global constants
SEQUENCE_LENGTH = 2000
NUM_CLASSES = 19

# =============================================================================================
# Parameters

# Data loading parameters
# tf.flags.DEFINE_float("val_sample_percentage", 0.1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_data_dir", "../../processed_data/word/train/", "Directory of training set")
tf.flags.DEFINE_string("val_data_dir", "../../processed_data/word/val/", "Directory of validation set")
tf.flags.DEFINE_string("embedding_path", "../../embeddings/word-embedding-300d-mc5.npy", "Path of embedding lookup table")

# Model hyper-parameters
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: 2,3,4)")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 256)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate (default: 0.001)")
# tf.flags.DEFINE_integer("batch_size", 128, "Batch size (default: 128)")
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


def get_batch(data_dir, i):
    """Get the `i`th batch of data set."""
    batch_file = os.path.join(data_dir, "%d.npz" % i)
    batch = np.load(batch_file)
    X_batch, y_batch = batch['X'], batch['y']
    return X_batch, y_batch


# def val_epoch(val_data_dir):
#     """Evaluate model on the validation set."""
#     batches_val = os.listdir(val_data_dir)
#     num_val_batches = len(batches_val)
#     loss = 0.0
#     labels_pred = list()
#     labels_true = list()
#     for i in range(num_val_batches):
#         X_batch, y_batch = get_batch(val_data_dir, i)


def main(argv=None):
    print("[INFO] Load embedding lookup table...")
    embedding_lookup_table = np.load(FLAGS.embedding_path)
    print("[INFO] Load finished!")

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # ================================================================================ #
            # 1. Assemble our graph                                                            #
            # ================================================================================ #

            print("[INFO] Assemble our graph...")

            cnn = TextCNN(sequence_length=SEQUENCE_LENGTH,
                          num_classes=NUM_CLASSES,
                          embedding_lookup_table=embedding_lookup_table,
                          filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                          num_filters=FLAGS.num_filters,
                          l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define training procedure
            global_step = tf.Variable(0, trainable=False, name="global_step")
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            print("[INFO] Assemble finished!")

            # ================================================================================ #
            # 2. Define `FileWriter` and `Saver` for saving summaries and models               #
            # ================================================================================ #

            # Output directories for summaries and models
            print("[INFO] Create a directory for saving summaries and models")
            timestamp = str(int(time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("[INFO] Write to {}".format(out_dir))

            # Keep track of gradient values and sparsity (optional)
            print("[INFO] Summaries for gradient values and sparsity")
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
            print("[INFO] Summaries for loss and accuracy")
            loss_summary = tf.summary.scalar(name="loss", tensor=cnn.loss)
            acc_summary = tf.summary.scalar(name="accuracy", tensor=cnn.accuracy)

            # Train summaries
            print("[INFO] Create `FileWriter` for train summaries")
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validate summaries
            print("[INFO] Create `FileWriter` for validate summaries")
            val_summary_op = tf.summary.merge([loss_summary, acc_summary])
            val_summary_dir = os.path.join(out_dir, "summaries", "val")
            val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

            # Create checkpoint directory and saver
            # Note: TensorFlow assumes this directory already exists so we need to create it
            print("[INFO] Create checkpoint directory and saver")
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # ================================================================================ #
            # 3. Execute our operations                                                        #
            # ================================================================================ #

            def train_step(X_batch, y_batch):
                """A single training step."""
                fetches = [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy]
                feed_dict = {cnn.X: X_batch, cnn.Y: y_batch, cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
                _, step, summaries, loss, accuracy = sess.run(fetches=fetches, feed_dict=feed_dict)
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("{} - step {:>6d}, loss {:8.5f}, acc {:4.2%}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, global_step=step)

            def val_step(X_batch, y_batch, writer=None):
                """Evaluate model on the validation set."""
                fetches = [global_step, val_summary_op, cnn.loss, cnn.accuracy]
                feed_dict = {cnn.X: X_batch, cnn.Y: y_batch, cnn.dropout_keep_prob: 1.0}
                step, summaries, loss, accuracy = sess.run(fetches=fetches, feed_dict=feed_dict)
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("{} - step {:>6d}, loss {:8.5f}, acc {:4.2%}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, global_step=step)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Set random number seed
            np.random.seed(42)

            batches_train = os.listdir(FLAGS.train_data_dir)
            batches_val = os.listdir(FLAGS.val_data_dir)
            num_train_batches = len(batches_train)
            num_val_batches = len(batches_val)

            print("[INFO] Start training...")

            for epoch in range(FLAGS.num_epochs):
                train_batch_indices_shuffled = np.random.permutation(num_train_batches)
                for i in range(num_train_batches):
                    train_batch = train_batch_indices_shuffled[i]
                    X_train, y_train = get_batch(FLAGS.train_data_dir, train_batch)
                    train_step(X_train, y_train)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        val_batch = np.random.randint(0, num_val_batches)
                        X_val, y_val = get_batch(FLAGS.val_data_dir, val_batch)
                        print("\n[INFO] Evaluation:")
                        val_step(X_val, y_val, val_summary_writer)
                        print()
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("[INFO] Save model checkpoint to {}\n".format(path))

            print("[INFO] Training finished! ( ^ _ ^ ) V")


if __name__ == '__main__':
    tf.app.run()
