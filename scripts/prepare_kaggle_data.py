import os
import tensorflow as tf
import matplotlib.image as mpimg

flags = tf.app.flags

flags.DEFINE_string('data_dir', None,
                    help='Path to image directory')

flags.DEFINE_string('tfrecord_filename', None,
                    'The output filename to name your TFRecord file')

FLAGS = flags.FLAGS

def convert_image(image_path):
    """
    Converts a png to a tensorflow example
    """
    image = mpimg.imread(image_path)
    example = tf.train.Example(features = tf.train.Features(feature = {
        'shape': tf.train.Feature(int64_list = tf.train.Int64List(value = image.shape)),
        'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image.tostring()]))
    }))
    return example

def main(argv):
    del argv;

    img_paths = os.listdir(FLAGS.data_dir)
    img_paths = [os.path.abspath(os.path.join(FLAGS.data_dir, i)) for i in img_paths]

    with tf.python_io.TFRecordWriter(FLAGS.tfrecord_filename) as writer:
        for img_path in img_paths:
            example = convert_image(img_path)
            writer.write(example.SerializeToString())

if __name__ == "__main__":
    tf.app.run()
