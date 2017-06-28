import tensorflow as tf
import os
import sys
sys.path.append('../../')
import data.raw_list_image as raw_list_image
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer

flags = tf.app.flags

flags.DEFINE_string('frozen_model_filename', '/your/model/path', 'Model Path.')
flags.DEFINE_string('model_name', 'resnet_28', 'Model Name.')
flags.DEFINE_string('data_list', '/your/test/data/list.txt',
                    'txt')
flags.DEFINE_integer('image_size', (112, 96), 'Image side length. (height, width)')
flags.DEFINE_integer('batch_size', 1, 'Test batch size')
flags.DEFINE_string('output_dir', '/workspace/datasets/LFW/intermediate/features',
                    'Directory to save output feature.')
flags.DEFINE_string('output_suffix', '.feature', 'feature formatted ouput name.')
flags.DEFINE_string('mode', 'test', 'test mode')

FLAGS = flags.FLAGS

def main(_):

    with tf.Graph().as_default() as graph:
        # Load freeze model
        output_graph_def = graph_pb2.GraphDef()

        with open(FLAGS.frozen_model_filename, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = importer.import_graph_def(output_graph_def, name="")

        x = graph.get_tensor_by_name('input:0')
        out = graph.get_tensor_by_name('unit_last/xw_plus_b:0')

        # Load test data list
        [path_list, _] = raw_list_image.get_path_label_list(FLAGS.data_list)

        with tf.Session(graph=graph) as sess:
            for path in path_list:
                path_tensor = tf.cast(path, tf.string)

                image_op = tf.read_file(path_tensor)
                image_op = tf.image.decode_jpeg(image_op, channels=3)
                image_op = tf.image.per_image_standardization(image_op)
                image_op = tf.reshape(image_op, [FLAGS.batch_size, FLAGS.image_size[0], FLAGS.image_size[1], 3])

                [image_val] = sess.run([image_op])
                features = sess.run(out, feed_dict={ x: image_val })

                [_, id, imgName] = path.rsplit('/', 2)

                dst = os.path.join(FLAGS.output_dir, id, imgName + '_' + FLAGS.model_name + '_1024'+ FLAGS.output_suffix)
                if not os.path.exists(os.path.dirname(dst)):
                    os.mkdir(os.path.dirname(dst))

                with open(dst, 'w') as fw:
                    fea_str = ' '.join(str(x) for x in features[0])
                    fw.write(fea_str)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
