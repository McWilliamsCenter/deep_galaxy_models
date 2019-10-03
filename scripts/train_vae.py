import tensorflow as tf
import tensorflow_hub as hub
import os
from absl import flags
from deepgal.nets import resnet_decoder, resnet_encoder
from deepgal.galsim import build_input_pipeline
from deepgal.VAEEstimator import vae_model_fn
from deepgal.flow import _clip_by_value_preserve_grad

# Model parameters
flags.DEFINE_integer("base_depth", default=128,
                     help="Base depth of the convolutional networks")

flags.DEFINE_integer("num_stages", default=1,
                     help="Number of residual network stages")

flags.DEFINE_integer("latent_size", default=64,
                     help="Number of dimensions in the latent code (z).")

flags.DEFINE_string("activation", default="leaky_relu",
                     help="Activation function for all hidden layers.")

flags.DEFINE_string("loglikelihood", default="Fourier",
                     help="Define in which space to compute the likelihood of the data, 'Fourier' or 'Pixel'")

flags.DEFINE_float("range_compression", default=0.003*20,
                     help="Apply arcsinh range compression to the images."
                     "Default is based on 20x noise standard deviation on COSMOS images at native resolution"
                     "Set to negative value to disable")

# Training parameters
flags.DEFINE_integer("batch_size", default=128,
                     help="Batch size.")

flags.DEFINE_float("learning_rate", default=0.0001,
                     help="Initial learning rate.")

flags.DEFINE_float("adam_epsilon", default=0.1,
                     help="Epsilon fuzz factor in ADAM optimizer.")

flags.DEFINE_float("gradient_clipping", default=10.,
                     help="Gradient norm clipping")

flags.DEFINE_float("kl_weight", default=0.001,
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

def make_decoder(base_depth, num_stages, activation, latent_size, range_compression):
    def decoder_fn(code, is_training):
        images = resnet_decoder(code, is_training=is_training, base_depth=base_depth, num_stages=num_stages,
                                activation=activation, scope='decoder')
        return images
    return decoder_fn

def make_encoder(base_depth, num_stages, activation, latent_size, range_compression):
    def encoder_fn(images, is_training):
        # Apply range compression
        if range_compression > 0:
            images = tf.asinh(images / range_compression)*range_compression

        code = resnet_encoder(images, is_training=is_training, base_depth=base_depth, num_stages=num_stages,
                            activation=activation, latent_size=latent_size, scope='encoder')
        return code
    return encoder_fn

def make_loglikelihood_fn(type):
    if type == 'Fourier':
        def loglikelihood_fn(xin, yin, features):
            x = tf.spectral.rfft2d(xin[...,0]) / tf.complex(tf.sqrt(tf.exp(features['ps'])),0.)
            y = tf.spectral.rfft2d(yin[...,0])  * features['psf'] / tf.complex(tf.sqrt(tf.exp(features['ps'])),0.)

            # Compute FFT normalization factor
            size = xin.get_shape().as_list()[1]
            pz = tf.reduce_sum(tf.abs(x - y)**2, axis=[-1, -2]) / size**2
            return -pz
    elif type == 'Pixel':
        def loglikelihood_fn(xin, yin, features):
            y = tf.spectral.irfft2d(tf.spectral.rfft2d(yin[...,0]) * features['psf'])
            pz = tf.reduce_sum(tf.abs(xin[:,:,:,0] - y)**2, axis=[-1, -2]) / features['nstd']**2
            return -pz
    else:
        raise NotImplemented()

    return loglikelihood_fn

def main(argv):
    del argv  # unused

    params = FLAGS.flag_values_dict()
    params["activation"] = getattr(tf.nn, params["activation"])
    params["loglikelihood_fn"] = make_loglikelihood_fn(FLAGS.loglikelihood)
    params["encoder_fn"] = make_encoder(FLAGS.base_depth, FLAGS.num_stages,
                                        params['activation'], FLAGS.latent_size, FLAGS.range_compression)
    params["decoder_fn"] = make_decoder(FLAGS.base_depth, FLAGS.num_stages,
                                        params['activation'], FLAGS.latent_size, FLAGS.range_compression)
    params['iaf_size'] = [[512, 512], [512, 512], [512, 512], [512, 512]]

    tf.gfile.MakeDirs(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.cache_dir)

    input_fn = build_input_pipeline(**params)

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
