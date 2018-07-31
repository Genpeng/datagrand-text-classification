# _*_ coding: utf-8 _*_

"""
Tensorflow implement of a CNN model for datagrand text process competition.

Author: StrongXGP <xgp1227#gmail.com>
Date:   2018/07/30
"""

import tensorflow as tf


class TextCNN:
    """A CNN model for text classification.

    The model structure is as followed:
    embedding layer -> convolution layer -> max pooling layer -> softmax layer
    """
    def __init__(self, sequence_length, num_classes, embedding_lookup_table,
                 filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_X = tf.placeholder(tf.int32, shape=[None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, shape=[None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.name_scope("embedding"):
            W_embedding = tf.get_variable(name="embedding_lookup_table",
                                          shape=embedding_lookup_table.shape,
                                          initializer=tf.constant_initializer(embedding_lookup_table),
                                          trainable=True)
            self.embedding_words = tf.nn.embedding_lookup(W_embedding, self.input_X)
            self.embedding_words_expanded = tf.expand_dims(self.embedding_words, axis=-1)

        # Create a convolution layer followed by a max pooling layer for each filter size
        pooled_outputs = list()
        for filter_size in filter_sizes:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution layer
                filter_shape = [filter_size, W_embedding.shape[1], 1, num_filters]
                W_filter = tf.get_variable(name="W_filter",
                                           shape=filter_shape,
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
                b_filter = tf.get_variable(name="b_filter",
                                           shape=[num_filters],
                                           initializer=tf.constant_initializer(0.1))
                conv = tf.nn.conv2d(input=self.embedding_words_expanded,
                                    filter=W_filter,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b_filter), name="relu")  # apply non-linearity

                # TODO: Add batch normalization

                # Max pooling layer
                pooled = tf.nn.max_pool(value=h,
                                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="max-pooling")

                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = len(filter_sizes) * num_filters
        self.h_pooled = tf.concat(pooled_outputs, axis=3)
        self.h_pooled_flat = tf.reshape(self.h_pooled, shape=[-1, num_filters_total])

        # TODO: Can add fully connected layer with batch normalization and dropout

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(x=self.h_pooled_flat, keep_prob=self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            W_out = tf.get_variable(name="W_out",
                                    shape=[num_filters_total, num_classes],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_out = tf.get_variable(name="b_out",
                                    shape=[num_classes],
                                    initializer=tf.constant_initializer(0.1))
            l2_loss += tf.nn.l2_loss(W_out)
            l2_loss += tf.nn.l2_loss(b_out)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W_out, b_out, name="scores")
            self.predictions = tf.argmax(self.scores, axis=1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, dtype=tf.float32), name="accuracy")
