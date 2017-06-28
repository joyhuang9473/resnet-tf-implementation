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

"""
"""

import tensorflow as tf
import os

def get_path_label_list(path_label_file):
  path_list = []
  label_list = []

  with open(path_label_file, 'r') as fr:
    lines = fr.readlines()

    for line in lines:
      line = line.strip()
      [path, label] = line.split(' ')
      label = int(label)

      # check path exist or not
      if not os.path.isfile(path):
        continue

      path_list.append(path)
      label_list.append(label)

  return path_list, label_list

def build_input(path_list, label_list, image_H, image_W, batch_size, mode):
  """Build [DATASET_NAME] image and labels.

  Args:
    path_list:
    label_list:
    image_H:
    image_W:
    batch_size:
    mode: Either 'train' or 'eval'.
  Returns:
    images: Batches of images. [batch_size, image_size, image_size, 3]
    labels: Batches of labels. [batch_size, num_classes]
  Raises:
    ValueError: when the specified dataset is not supported.
  """
  path_tensor_list = tf.cast(path_list, tf.string)
  label_tensor_list = tf.cast(label_list, tf.int32)
  depth = 3 # image color channel

  # make an input queue
  input_queue = tf.train.slice_input_producer([path_tensor_list, label_tensor_list])
  image_contents = tf.read_file(input_queue[0])
  image = tf.image.decode_jpeg(image_contents, channels=3)
  label = tf.reshape(input_queue[1], [1])

  if mode == 'train':
    #=====
    # Data augmentation
    #=====
    # image = tf.image.resize_image_with_crop_or_pad(
        # image, image_H+4, image_W+4)
    # image = tf.random_crop(image, [image_H, image_W, 3])
    # image = tf.image.random_flip_left_right(image)
    # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
    # image = tf.image.random_brightness(image, max_delta=63. / 255.)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.per_image_standardization(image)

    example_queue = tf.RandomShuffleQueue(
        capacity=16 * batch_size,
        min_after_dequeue=8 * batch_size,
        dtypes=[tf.float32, tf.int32],
        shapes=[[image_H, image_W, depth], [1]])
    num_threads = 16
  else:
    # image = tf.image.resize_image_with_crop_or_pad(
        # image, image_H, image_W)
    image = tf.image.per_image_standardization(image)

    example_queue = tf.FIFOQueue(
        3 * batch_size,
        dtypes=[tf.float32, tf.int32],
        shapes=[[image_H, image_W, depth], [1]])
    num_threads = 1

  example_enqueue_op = example_queue.enqueue([image, label])
  tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
      example_queue, [example_enqueue_op] * num_threads))

  # Read 'batch' labels + images from the example queue.
  images, labels = example_queue.dequeue_many(batch_size)
  labels = tf.reshape(labels, [batch_size])

  assert len(images.get_shape()) == 4
  assert images.get_shape()[0] == batch_size
  assert images.get_shape()[-1] == 3
  assert len(labels.get_shape()) == 1
  assert labels.get_shape()[0] == batch_size

  # Display the training images in the visualizer.
  tf.summary.image('images', images)
  return images, labels
