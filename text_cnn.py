# _*_ coding: utf-8 _*_

"""A convolutional neural network (CNN) for long text classification.

Reference:
- Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).

Author: StrongXGP <xgp1227@gmail.com>
Date:   2018/07/05
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

