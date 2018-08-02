# _*_ coding: utf-8 _*_

import gc
import os
import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

# Self-defined modules or classes
import data_helper
from text_cnn import TextCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
# =========================================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "../../raw_data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data")
tf.flags.DEFINE_string("negative_data_file", "../../raw_data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data")

# Model hyperparameter
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: 3,4,5)")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def preprocess():
    # Data preparation
    #======================================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helper.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # Build vocabulary
    max_document_length = max([len(x.split()) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * len(y))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x_text, x, y, x_shuffled, y_shuffled
    gc.collect()

    print("Vocabulary size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev


def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # Training
    #======================================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(sequence_length=x_train.shape[1],
                          num_classes=y_train.shape[1],
                          vocab_size=len(vocab_processor.vocabulary_),
                          embedding_size=FLAGS.embedding_dim,
                          filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                          num_filters=FLAGS.num_filters,
                          l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram(name="{}/grad/hist".format(v.name), values=g)
                    sparsity_summary = tf.summary.scalar(name="{}/grad/sparsity".format(v.name),
                                                         tensor=tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. TensorFlow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """A single training step."""
                feed_dict = {cnn.input_x: x_batch,
                             cnn.input_y: y_batch,
                             cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("{} - step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """Evaluates model on a dev set."""
                feed_dict = {cnn.input_x: x_batch,
                             cnn.input_y: y_batch,
                             cnn.dropout_keep_prob: 1.0}
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("{} - step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helper.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation: ")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print()
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    print("Generate training and validation data")
    t0_preprocess = time.time()
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    print("Done in %.3f seconds!" % (time.time() - t0_preprocess))
    print()

    print("Start training...")
    t0_train = time.time()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)
    print("Done in %.3f seconds!" % (time.time() - t0_train))
    print("Training finished! ( ^ _ ^ ) V")


if __name__ == '__main__':
    tf.app.run()
