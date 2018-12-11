from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow_probability as tfp
import tensorflow_hub as hub
tfd = tfp.distributions
tfb = tfp.bijectors

__all__ = ['VAEEstimator', 'vae_model_fn']


def _softplus_inverse(x):
    """Helper which computes the function inverse of `tf.nn.softplus`."""
    return tf.log(tf.math.expm1(x))

def make_encoder_spec(encoder_fn, n_channels, image_size, latent_size, iaf_size, is_training):
    # Create a module for the encoding task
    def encoder_module_fn():
        input_layer = tf.placeholder(tf.float32, shape=[None, image_size, image_size, n_channels])
        sample_shape = tf.placeholder(tf.int32)

        net = encoder_fn(input_layer, latent_size=latent_size, is_training=is_training, scope='encoder')

        encoding = tfd.MultivariateNormalDiag(
            loc=net[..., :latent_size],
            scale_diag=tf.nn.softplus(net[..., latent_size:] + _softplus_inverse(1.0)),
            name="code")

        # Use IAF for modeling the approximate posterior
        chain = []
        def get_permutation(name):
            return tf.get_variable(name, initializer=np.random.permutation(latent_size).astype("int32"), trainable=False)
        for i,s in enumerate(iaf_size):
            chain.append(tfb.Invert(tfb.MaskedAutoregressiveFlow(
                        shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                            hidden_layers=s))))
            chain.append(tfb.Permute(permutation=get_permutation(name='permutation_%d'%i)))

        iaf = tfd.TransformedDistribution(
                    distribution=encoding,
                    bijector=tfb.Chain(chain))

        sample = iaf.sample(sample_shape)
        log_prob = iaf.log_prob(sample)
        hub.add_signature(inputs={'image': input_layer, 'sample_shape': sample_shape},
                          outputs={'sample': sample, 'log_prob': log_prob})

    return hub.create_module_spec(encoder_module_fn)


def make_decoder_spec(decoder_fn, latent_size, is_training):
    # Module for the decoding task, returns an unconvolved light profile
    def decoder_module_fn():
        code = tf.placeholder(tf.float32, shape=[None, latent_size])
        net = decoder_fn(code, is_training=is_training, scope='generator')
        hub.add_signature(inputs=code, outputs=net)

    return hub.create_module_spec(decoder_module_fn)


def pack_images(images, rows, cols):
    """Helper utility to make a field of images."""
    shape = tf.shape(images)
    width = shape[-3]
    height = shape[-2]
    depth = shape[-1]
    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(images)[0]
    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)
    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    images = tf.reshape(images, [1, rows * width, cols * height, depth])
    return images


def image_tile_summary(name, tensor, rows=8, cols=8):
    tf.summary.image(name, pack_images(tensor, rows, cols), max_outputs=1)


def vae_model_fn(features, labels, mode, params, config):
    """
    Model function to create a VAE estimator
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Extract input images
    x = features['x']

    # Build components of the model
    encoder_spec = make_encoder_spec(params['encoder_fn'], x.shape[-1],
                                     x.shape[-2],
                                     params['latent_size'],
                                     params['iaf_size'], is_training)
    encoder = hub.Module(encoder_spec, name='encoder', trainable=True)
    decoder_spec = make_decoder_spec(params['decoder_fn'],
                                     params['latent_size'],
                                     is_training)
    decoder = hub.Module(decoder_spec, name='decoder', trainable=True)
    prior = tfd.MultivariateNormalDiag(
                loc=tf.zeros([latent_size]),
                scale_identity_multiplier=1.0)

    # Sample from the infered posterior
    code = encoder({'image': x, 'sample_shape': params['sample_shape']}, as_dict=True)
    code_shape = tf.shape(code['sample'])
    recon = decoder(tf.reshape(code['sample'], (-1, params['latent_size'])))
    if params['sample_shape'] > 1:
        code_shape = tf.shape(code['sample'])
        recon = tf.reshape(recon, [code_shape[0], code_shape[1],
                                   recon.shape[-3], recon.shape[-2],
                                   recon.shape[-1]])

    image_tile_summary("image", tf.to_float(x[:16]), rows=4, cols=4)
    image_tile_summary("recon", tf.to_float(recon[0, :16]), rows=4, cols=4)
    image_tile_summary("diff", tf.to_float(x[:16] - recon[0, :16]), rows=4, cols=4)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Register and export encoder and decoder modules
        hub.register_module_for_export(encoder, "encoder")
        hub.register_module_for_export(decoder, "decoder")

        # In case of prediction, we only sample new images from the prior
        z = prior.sample(params['sample_shape'])
        image = decoder(z)
        predictions = {'code': z, 'image': image}
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    # This is the loglikelihood of a batch of images
    loglikelihood = params['loglikelihood_fn'](x, recon, features)
    tf.summary.scalar('loglikelihood', tf.reduce_mean(loglikelihood))

    kl = code['log_prob'] - prior.log_prob(code['sample'])
    tf.summary.scalar('kl', tf.reduce_mean(kl))

    elbo = loglikelihood - kl

    loss = - tf.reduce_mean(elbo)
    tf.summary.scalar("elbo", tf.reduce_mean(elbo))

    importance_weighted_elbo = tf.reduce_mean(
      tf.reduce_logsumexp(elbo, axis=0) - tf.log(tf.to_float(params['sample_shape'])))
    tf.summary.scalar("elbo/importance_weighted", importance_weighted_elbo)

    # Randomly samples a bunch of examples for visualisation
    random_code = prior.sample(16)
    random_image = decoder(random_code)
    image_tile_summary("random", tf.to_float(random_image), rows=4, cols=4)

    # Training of the model
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.cosine_decay(params["learning_rate"], global_step,
                                          params["max_steps"])
    tf.summary.scalar("learning_rate", learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)

    eval_metric_ops = {
        "elbo/importance_weighted": tf.metrics.mean(importance_weighted_elbo),
        "elbo": tf.metrics.mean(tf.reduce_mean(elbo)),
        "kl": tf.metrics.mean(tf.reduce_mean(kl)),
        "loglikelihood": tf.metrics.mean(tf.reduce_mean(loglikelihood))}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


class VAEEstimator(tf.estimator.Estimator):
    """An estimator for Vanilla Variational Auto-Encoders (VAEs).
    """

    def __init__(self,
                 encoder_fn=None,
                 decoder_fn=None,
                 loglikelihood_fn=None,
                 latent_size=16,
                 sample_shape=16,
                 iaf_size=[[256,256],[256,256]],
                 learning_rate=0.001,
                 max_steps=5001,
                 model_dir=None, config=None):
        """
        Args:
            encoder_fn: model function for the encoder
        """
        params = {}
        params['encoder_fn'] = encoder_fn
        params['decoder_fn'] = decoder_fn
        params['loglikelihood_fn'] = loglikelihood_fn
        params['latent_size'] = latent_size
        params['sample_shape'] = sample_shape
        params['iaf_size'] = iaf_size
        params['learning_rate'] = learning_rate
        params['max_steps'] = max_steps

        super(self.__class__, self).__init__(model_fn=vae_model_fn,
                                             model_dir=model_dir,
                                             params=params,
                                             config=config)
