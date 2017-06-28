# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet model.
Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""

import numpy as np
import tensorflow as tf

class BasicModel(object):
  def __init__(self, hps, mode):
    """ResNet constructor.
    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self.mode = mode
    self._extra_train_ops = []

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _residual(self, x, in_filter, out_filter, stride):
    """Residual unit with 2 sub layers."""
    with tf.variable_scope('residual_only_activation'):
      orig_x = x

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)
      x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub2'):
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.summary.histogram(var.op.name, var)

    return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.contrib.layers.xavier_initializer())
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    shape = x.get_shape().as_list()
    in_dim = shape[1]*shape[2]*shape[3] if len(shape) == 4 else shape[1]

    x = tf.reshape(x, [-1, in_dim])
    w = tf.get_variable(
        'DW', [in_dim, out_dim],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    # TODO
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])
