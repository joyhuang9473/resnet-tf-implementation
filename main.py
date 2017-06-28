import tensorflow as tf
import models.resnet as resnet
import time
import os
import sys
from collections import namedtuple

import data.raw_list_image as raw_list_image
from data.casia_webface_data import CasiaWebFaceData

flags = tf.app.flags

flags.DEFINE_string('model_name', 'resnet-28', 'Model Name.')
flags.DEFINE_string('mode', 'train', 'train or validation.')
flags.DEFINE_string('dataset', 'CASIA-WebFace',
                    'dataset name. [CASIA-WebFace]')
flags.DEFINE_string('data_list', '/workspace/datasets/TRAIN_SET_D1/intermediate/train.txt',
                    'txt')
flags.DEFINE_integer('image_size', (112, 96), 'Image side length. (height, width)')
flags.DEFINE_string('snapshot_dir', './results/train',
                    'Directory to keep training outputs.')
flags.DEFINE_string('resume_model_dir', '',
                    'Directory to resume training outputs.')
flags.DEFINE_string('log_dir', './results/train',
                    'Directory to keep logs.')
flags.DEFINE_integer('max_steps', 1000000,
                     'Number of (global) training steps to perform')
flags.DEFINE_integer('batch_size', 256, 'Training batch size')
flags.DEFINE_integer('display', 100, 'Display log message')
flags.DEFINE_integer('snapshot', 1000, 'Snapshpt model')
flags.DEFINE_float('learning_rate', 0.01, 'Learning Rate')
flags.DEFINE_integer('devices', ['/gpu:0'],
                     'GPU or CPU')

FLAGS = flags.FLAGS
HParams = namedtuple('HParams',
                     'batch_size, num_classes, lrn_rate, '
                     'weight_decay_rate, relu_leakiness, optimizer')

class _LearningRateSetterHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""
    def __init__(self, model):
        tf.train.SessionRunHook.__init__(self)
        self.model = model

    def begin(self):
        self._lrn_rate = FLAGS.learning_rate

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            self.model.global_step,  # Asks for global step value.
            feed_dict={self.model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def after_run(self, run_context, run_values):
        train_step = run_values.results
        if train_step < 20000:
            self._lrn_rate = 0.01
        elif train_step < 30000:
            self._lrn_rate = 0.001
        elif train_step < 50000:
            self._lrn_rate = 0.0001
        else:
            self._lrn_rate = 0.00001

def train(hps):
    """Prepare data"""
    [path_list, label_list] = raw_list_image.get_path_label_list(FLAGS.data_list)
    [images_op, labels_op] = raw_list_image.build_input(path_list,
                                                        label_list,
                                                        FLAGS.image_size[0],
                                                        FLAGS.image_size[1],
                                                        FLAGS.batch_size,
                                                        FLAGS.mode)

    """Prepare model graph"""
    _images = tf.placeholder(tf.float32, shape=[None, FLAGS.image_size[0], FLAGS.image_size[1], 3], name='input')
    _labels = tf.placeholder(tf.int32, shape=[None], name='label')

    model = resnet.ResNet(hps, _images, _labels, FLAGS.mode)
    model.build_graph()

    """Print total parameters"""
    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
                    tf.get_default_graph(),
                    tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

    """Print model analysis"""
    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    """Prepare saver"""
    saver = tf.train.Saver()
    saverHook = tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.snapshot_dir,
                                             save_secs=None,
                                             save_steps=FLAGS.snapshot,
                                             saver=saver,
                                             checkpoint_basename=FLAGS.model_name + '.ckpt')

    lrHook = _LearningRateSetterHook(model)

    with tf.train.MonitoredTrainingSession(
        hooks=[lrHook, saverHook],
        save_checkpoint_secs=None,
        save_summaries_steps=None,
        config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        """Prepare summary writer"""
        train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        if FLAGS.resume_model_dir:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(FLAGS.resume_model_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)

        step = sess.run(model.global_step)

        while step < FLAGS.max_steps:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

            start_time = time.time()

            [images, labels] = sess.run([images_op, labels_op])
            _, loss_value, summaries, step = sess.run([model.train_op, model.cost, model.summaries, model.global_step],
                                                feed_dict = { _images: images,
                                                              _labels: labels })
            duration = time.time() - start_time

            if step % FLAGS.display == 0:
                num_examples_per_step = FLAGS.batch_size * len(FLAGS.devices)
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / len(FLAGS.devices)

                format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

                train_writer.add_summary(summaries, step)
            if step == FLAGS.max_steps:
                train_writer.add_summary(summaries, step)

        train_writer.close()

        print('Finish training!')

def validation(hps):
    print('TODO')

def main(_):

    assert FLAGS.dataset in ['CASIA-WebFace'], (
        'Please make the FLAGS.dataset commensurate with dataset list.')

    if FLAGS.dataset == 'CASIA-WebFace':
        dataset = CasiaWebFaceData(subset=FLAGS.mode)

    # TODO: tfrecord fail
    # assert dataset.data_files()

    hps = HParams(batch_size=FLAGS.batch_size,
                  num_classes=dataset.num_classes(),
                  lrn_rate=FLAGS.learning_rate,
                  weight_decay_rate=0.0002,
                  relu_leakiness=0.1,
                  optimizer='sgd')

    with tf.Graph().as_default():
        for dev in FLAGS.devices:
            with tf.device(dev):
                if FLAGS.mode == 'train':
                    train(dataset, hps)
                elif FLAGS.mode == 'validation':
                    validation(dataset, hps)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
