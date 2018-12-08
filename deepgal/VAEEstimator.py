from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow_probability as tfp
import tensorflow_hub as hub
tfd = tensorflow_probability.distributions

__all__ = ['VAEEstimator']

def _softplus_inverse(x):
  """Helper which computes the function inverse of `tf.nn.softplus`."""
  return tf.log(tf.math.expm1(x))

def _make_mixture_prior(latent_size, mixture_components):
  """Creates the mixture of Gaussians prior distribution.
  Args:
    latent_size: The dimensionality of the latent representation.
    mixture_components: Number of elements of the mixture.
  Returns:
    random_prior: A `tfd.Distribution` instance representing the distribution
      over encodings in the absence of any evidence.
  """
  if mixture_components == 1:
    # See the module docstring for why we don't learn the parameters here.
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros([latent_size]),
        scale_identity_multiplier=1.0)

  loc = tf.get_variable(name="loc", shape=[mixture_components, latent_size])
  raw_scale_diag = tf.get_variable(
      name="raw_scale_diag", shape=[mixture_components, latent_size])
  mixture_logits = tf.get_variable(
      name="mixture_logits", shape=[mixture_components])

  return tfd.MixtureSameFamily(
      components_distribution=tfd.MultivariateNormalDiag(
          loc=loc,
          scale_diag=tf.nn.softplus(raw_scale_diag)),
      mixture_distribution=tfd.Categorical(logits=mixture_logits),
      name="prior")



class VAEEstimator(tf.estimator.Estimator):
    """An estimator for Vanilla Variational Auto-Encoders (VAEs).
    """

    def __init__(self,
                 latent_size,
                 encoder_fn=None,
                 decoder_fn=None,
                 reconstruction_loss_fn=None,
                 learning_rate=0.001,
                 model_dir=None, config=None):
        """
        Args:
            encoder_fn: model function for the encoder
        """

        def _model_fn(features, labels, mode):
            return _vae_model_fn(latent_size, features, labels, mode, encoder_fn, decoder_fn,
                                 reconstruction_loss_fn, learning_rate, config)

        super(VAEEstimator, self).__init__(model_fn=_model_fn,
                                           model_dir=model_dir,
                                           config=config)


def _vae_model_fn(latent_size, features, labels, mode, encoder_fn, decoder_fn,
                  reconstruction_loss_fn, learning_rate, config):

    is_training = (mode == tf.estimator.ModeKeys)

    x = features['x']

    # Create a module for the encoding task
    def encoder_module_fn():
        input_layer = tf.placeholder(tf.float32, shape=[None] + x.shape[1:])
        sample_shape = tf.placeholder(tf.int32)

        net = encoder_fn(x, latent_size=latent_size, is_training=is_training, scope='encoder')
        encoding=tfd.MultivariateNormalDiag(
            loc=net[..., :latent_size],
            scale_diag=tf.nn.softplus(net[..., latent_size:] + _softplus_inverse(1.0)),
            name="code")

        sample = tf.squeeze(encoding.sample(sample_shape))
        log_prob = encoding.log_prob(sample)
        hub.add_signature(inputs={'features':input_layer, 'sample_shape':sample_shape},
                          outputs={'sample':sample, 'log_prob':log_prob})

    encoder_spec = hub.create_module_spec(encoder_module_fn)
    encoder_module = hub.Module(encoder_spec, name='encoder',trainable=True)

    # Module for the decoding task, returns an unconvolved light profile
    def decoder_module_fn():
        code = tf.placeholder(tf.float32, shape=[None, latent_size])
        net = decoder_fn(code, is_training=is_training, scope='generator')
        hub.add_signature(inputs=code, outputs=net)

    decoder_spec = hub.create_module_spec(decoder_module_fn)
    decoder_module = hub.Module(decoder_spec, name='decoder', trainable=True)

    # Sample from the infered posterior
    code  = encoder_module({'features':features, 'sample_size':1})
    recon = decoder_module(code)

    # This is the loglikelihood of a batch of images
    loglikelihood = reconstruction_loss_fn(labels, recon, features)

    kl = 


    train_op = None
    eval_metric_ops = None
    if mode == tf.estimator.ModeKeys.TRAIN:

        tf.summary.image('input', features)
        tf.summary.image('rec', rec)
        tf.summary.image('diff', features - recon )

        # Compute KL divergence between code and prior distribution
        kl = tf.reduce_mean(tf.reduce_sum( tf.clip_by_value(kl_normal2_stdnormal(qz_mu, qz_logvar),1.,100),axis=-1))
        tf.summary.scalar('kl_divergence', kl)
        tf.losses.add_loss(kl)

        rec_loss = reconstruction_loss_fn(labels, predictions, features)
        tf.summary.scalar('reconstruction', rec_loss)

        total_loss = tf.losses.get_total_loss()
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=total_loss,
                                                                                global_step=tf.train.get_global_step())
    elif mode == tf.estimator.ModeKeys.EVAL:

        eval_metric_ops = {
            "kl": kl_normal2_stdnormal(qz_mu, qz_logvar).sum(axis=-1).mean(),
            "log_p(x|z)": reconstruction_loss_fn(labels, predictions, features).mean()
        }

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)
