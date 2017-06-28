import tensorflow as tf
import basic_model
BasicModel = basic_model.BasicModel

class ResNet(BasicModel):
  """ResNet model."""
  def __init__(self, hps, images, labels, mode):
    BasicModel.__init__(self, hps, mode)

    self._images = images
    self._labels = labels

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.contrib.framework.get_or_create_global_step()
    self.build_model()
    if self.mode == 'train':
      self.build_train_op()
    self.summaries = tf.summary.merge_all()

  def build_model(self):
    """Build the core model within the graph."""
    with tf.variable_scope('init'):
      x = self._images
      x = self._conv('init_conv', x, 3, 3, 32, self._stride_arr(1))
      x = self._relu(x, self.hps.relu_leakiness)

    # ==========
    # Block 1
    # ==========
    with tf.variable_scope('unit_1_0'):
      x = self._conv('conv1', x, 3, 32, 64, self._stride_arr(1))
      x = self._relu(x, self.hps.relu_leakiness)
      x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('unit_1_1'):
      x = self._residual(x, 64, 64, self._stride_arr(1))

    # ==========
    # Block 2
    # ==========
    with tf.variable_scope('unit_2_0'):
      x = self._conv('conv1', x, 3, 64, 128, self._stride_arr(1))
      x = self._relu(x, self.hps.relu_leakiness)
      x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('unit_2_1'):
      x = self._residual(x, 128, 128, self._stride_arr(1))

    with tf.variable_scope('unit_2_2'):
      x = self._residual(x, 128, 128, self._stride_arr(1))

    # ==========
    # Block 3
    # ==========
    with tf.variable_scope('unit_3_0'):
      x = self._conv('conv1', x, 3, 128, 256, self._stride_arr(1))
      x = self._relu(x, self.hps.relu_leakiness)
      x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('unit_3_1'):
      x = self._residual(x, 256, 256, self._stride_arr(1))

    with tf.variable_scope('unit_3_2'):
      x = self._residual(x, 256, 256, self._stride_arr(1))

    with tf.variable_scope('unit_3_3'):
      x = self._residual(x, 256, 256, self._stride_arr(1))

    with tf.variable_scope('unit_3_4'):
      x = self._residual(x, 256, 256, self._stride_arr(1))

    with tf.variable_scope('unit_3_5'):
      x = self._residual(x, 256, 256, self._stride_arr(1))

    # ==========
    # Block 4
    # ==========
    with tf.variable_scope('unit_4_0'):
      x = self._conv('conv1', x, 3, 256, 512, self._stride_arr(1))
      x = self._relu(x, self.hps.relu_leakiness)
      x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('unit_4_1'):
      x = self._residual(x, 512, 512, self._stride_arr(1))

    with tf.variable_scope('unit_4_2'):
      x = self._residual(x, 512, 512, self._stride_arr(1))

    with tf.variable_scope('unit_4_3'):
      x = self._residual(x, 512, 512, self._stride_arr(1))

    with tf.variable_scope('unit_last'):
      x = self._fully_connected(x, 512)

    with tf.variable_scope('logit'):
      logits = self._fully_connected(x, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
      xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=self._labels)
      self.cost = tf.reduce_mean(xent, name='xent')
      self.cost += self._decay()

      tf.summary.scalar('cost', self.cost)

  def build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.summary.scalar('learning_rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    if self.hps.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)
