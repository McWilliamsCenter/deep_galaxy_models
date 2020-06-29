import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import os
from absl import flags
from deepgal.nets import resnet_decoder, resnet_encoder
from deepgal.VAEEstimator import vae_model_fn

# Dataset parameters
flags.DEFINE_string("data_filename", default='/data2/KaggleGalaxyZoo/images_training2.tfrecord',
                    help='Path to tfrecord file')

flags.DEFINE_integer("stamp_size", default=64,
                    help="Size of the postage stamps")

# Model parameters
flags.DEFINE_integer("base_depth", default=128,
                     help="Base depth of the convolutional networks")

flags.DEFINE_integer("num_stages", default=1,
                     help="Number of residual network stages")

flags.DEFINE_integer("latent_size", default=100,
                     help="Number of dimensions in the latent code (z).")

flags.DEFINE_string("activation", default="leaky_relu",
                     help="Activation function for all hidden layers.")

# Training parameters
flags.DEFINE_integer("batch_size", default=128,
                     help="Batch size.")

flags.DEFINE_float("learning_rate", default=0.0002,
                     help="Initial learning rate.")

flags.DEFINE_float("adam_epsilon", default=0.0001,
                     help="Epsilon fuzz factor in ADAM optimizer.")

flags.DEFINE_float("gradient_clipping", default=1.,
                     help="Gradient norm clipping")

flags.DEFINE_float("kl_weight", default=1.,
                     help="Weighting for the KL divergence constraint")

flags.DEFINE_integer("max_steps", default=250001,
                     help="Number of training steps to run.")

flags.DEFINE_string("model_dir", default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/"),
                     help="Directory to put the model's fit.")

flags.DEFINE_string("export_dir", default="modules/vae",
                     help="Directory to put the trained tensorflow modules.")

flags.DEFINE_integer("save_checkpoints_steps", default=1000,
                     help="Frequency at which to save checkpoints.")

FLAGS = flags.FLAGS

def make_decoder(base_depth, num_stages, activation, latent_size):
    def decoder_fn(code, is_training):
        images = resnet_decoder(code, is_training=is_training, base_depth=base_depth, num_stages=num_stages,
                                output_channels=3*8*3,
                                activation=activation, scope='decoder')

        # We are describing the image in terms of a mixture of  discretized logistic
        # distributions, as in Salimans et al
        r,g,b = tf.split(images, num_or_size_splits=3, axis=-1)
        images = tf.stack([r,g,b], axis=3)
        loc, scale, logits = tf.split(images, num_or_size_splits=3, axis=-1)
        scale = tf.nn.softplus(scale)

        discretized_logistic_dist = tfd.QuantizedDistribution(
                    distribution=tfd.TransformedDistribution(
                        distribution=tfd.Logistic(loc=loc, scale=scale),
                        bijector=tfb.AffineScalar(shift=-0.5)),
                    low=0.,
                    high=2**8 - 1.)
        dist = tfd.Independent(tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=discretized_logistic_dist))
        return dist

    return decoder_fn

def make_encoder(base_depth, num_stages, activation, latent_size):
    def encoder_fn(images, is_training):
        # Standardize the images
        images = images / 256. - 0.5
        code = resnet_encoder(images, is_training=is_training, base_depth=base_depth, num_stages=num_stages,
                            activation=activation, latent_size=latent_size, scope='encoder')
        return code
    return encoder_fn

def make_input_fn(data_filename, stamp_size, batch_size, **kwargs):

    def parsing_function(tfrecord):
        features = {
            'image': tf.FixedLenFeature([], tf.string),
        }
        im = tf.parse_single_example(tfrecord, features)['image']
        im = tf.image.decode_image(im)
        im.set_shape([None, None, 3])
        im = tf.image.central_crop(im, 0.5)
        im = tf.image.resize_images(im,
                                    size=[stamp_size, stamp_size],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return im

    def augment(images):
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_flip_up_down(images)
        return images

    def input_fn():
        data = tf.data.TFRecordDataset([data_filename])
        data = data.map(parsing_function)
        data = data.cache().repeat()
        data = data.shuffle(10000)
        data = data.apply(tf.contrib.data.map_and_batch(
                map_func=augment, batch_size=batch_size, num_parallel_calls=6))
        data = data.prefetch(16)
        iterator = data.make_one_shot_iterator()
        batch_im = tf.to_float(iterator.get_next())
        return {'x': batch_im}, batch_im

    return input_fn

def main(argv):
    del argv  # unused

    params = FLAGS.flag_values_dict()
    params["activation"] = getattr(tf.nn, params["activation"])
    params["encoder_fn"] = make_encoder(FLAGS.base_depth, FLAGS.num_stages,
                                        params['activation'], FLAGS.latent_size)
    params["decoder_fn"] = make_decoder(FLAGS.base_depth, FLAGS.num_stages,
                                        params['activation'], FLAGS.latent_size)
    params['iaf_size'] = [[512, 512], [512, 512], [512, 512], [512, 512]]

    tf.gfile.MakeDirs(FLAGS.model_dir)

    input_fn = make_input_fn(**params)

    estimator = tf.estimator.Estimator(
      vae_model_fn,
      params=params,
      config=tf.estimator.RunConfig(
          model_dir=FLAGS.model_dir,
          save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      ))

    estimator.train(input_fn=input_fn, max_steps=FLAGS.max_steps)

    exporter = hub.LatestModuleExporter("tf_hub",
        tf.estimator.export.build_raw_serving_input_receiver_fn(input_fn()[0]))
    exporter.export(estimator, FLAGS.export_dir, estimator.latest_checkpoint())

if __name__ == "__main__":
    tf.app.run()
