# _*_ coding: utf-8 _*_

"""
A convolutional neural network (CNN) for long text classification.

Reference:
- http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
- Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).

Author: StrongXGP <xgp1227@gmail.com>
Date:   2018/07/05
"""

import tensorflow as tf


class TextCNN:
    """
    A CNN for text classification.
    The model structure is: embedding -> convolution -> max pooling -> softmax
    """

    def __init__(self, sequence_length, num_classes, vocab_size,             # parameters about data
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda):  # parameters about model

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, shape=[None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding_lookup_table = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                                      name="embedding_lookup_table")
            self.embedding_chars = tf.nn.embedding_lookup(self.embedding_lookup_table, self.input_x)
            self.embedding_chars_expanded = tf.expand_dims(self.embedding_chars, -1)

        # Create a convolution layer followed by max-pooling layer for each filter size
        pooled_outputs = []
        for filter_size in filter_sizes:
            with tf.name_scope("conv%s-maxpool" % filter_size):
                # Convolution layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                conv_kernel = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_kernel")
                conv_bias = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="conv_bias")
                conv = tf.nn.conv2d(input=self.embedding_chars_expanded,
                                    filter=conv_kernel,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, conv_bias), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(value=h,
                                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name="max-pooling")
                pooled_outputs.append(pooled)

        # Combined all the pooled features
        num_filters_total = len(filter_sizes) * num_filters
        self.h_pool = tf.concat(pooled_outputs, axis=3)
        self.h_pool_flat = tf.reshape(self.h_pool, shape=[-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            W = tf.get_variable(name="W",
                                shape=[num_filters_total, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name="b",
                                shape=[num_classes],
                                initializer=tf.constant_initializer(0.1))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, axis=1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
