from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from functools import partial
from absl import flags
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow_probability as tfp
import tensorflow_hub as hub
tfd = tfp.distributions
tfb = tfp.bijectors
from .flow import _clip_by_value_preserve_grad


__all__ = ['VAEEstimator', 'vae_model_fn']


def _softplus_inverse(x):
    """Helper which computes the function inverse of `tf.nn.softplus`."""
    return tf.log(tf.math.expm1(x))

def make_encoder_fn(encoder_fn, latent_size, iaf_size, is_training):
    def encoder_module_fn(images):
        net = encoder_fn(images, is_training=is_training)
        loc, scale  = tf.split(net, [latent_size, latent_size], axis=-1)

        encoding = tfd.MultivariateNormalDiag(
            loc=loc,
            scale_diag=tf.nn.softplus(scale + _softplus_inverse(1.0)),
            name="code")

        # Use IAF for modeling the approximate posterior
        chain = []
        def get_permutation(name):
            return tf.get_variable(name, initializer=np.random.permutation(latent_size).astype("int32"), trainable=False)
        for i,s in enumerate(iaf_size):
            chain.append(tfb.Invert(tfb.MaskedAutoregressiveFlow(
                            shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                            hidden_layers=s,
                            shift_only=True, name='maf%d'%i), is_constant_jacobian=True)))
            chain.append(tfb.Permute(permutation=get_permutation(name='permutation_%d'%i)))

        iaf = tfd.TransformedDistribution(
                    distribution=encoding,
                    bijector=tfb.Chain(chain))
        return iaf
    return encoder_module_fn


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

    # Build model functions
    encoder_model = make_encoder_fn(params['encoder_fn'],
                                    params['latent_size'],
                                    params['iaf_size'], is_training=is_training)
    decoder_model = partial(params['decoder_fn'], is_training=is_training)

    # Define latent prior
    prior = tfd.MultivariateNormalDiag(
                loc=tf.zeros([params['latent_size']]),
                scale_identity_multiplier=1.0)

    # In predict mode, we encapsulate the model inside a module for exporting
    # This is because of a weird bug that makes training MAF unstable inside a
    # module
    if mode == tf.estimator.ModeKeys.PREDICT:
        image_size = x.shape[-2]
        n_channels = x.shape[-1]
        latent_size = params['latent_size']
        def make_encoder_spec():
            input_layer = tf.placeholder(tf.float32, shape=[None, image_size, image_size, n_channels])
            encoding = encoder_model(input_layer)
            sample = encoding.sample()
            log_prob = encoding.log_prob(sample)
            hub.add_signature(inputs=input_layer,
                              outputs={'sample': sample, 'log_prob': log_prob})

        encoder_spec = hub.create_module_spec(make_encoder_spec)
        encoder = hub.Module(encoder_spec, name="encoder_module")

        def make_decoder_spec():
            code = tf.placeholder(tf.float32, shape=[None, latent_size])
            output = decoder_model(code)
            if not tf.contrib.framework.is_tensor(output):
                output = output.sample()
            hub.add_signature(inputs=code, outputs=output)

        decoder_spec = hub.create_module_spec(make_decoder_spec)
        decoder = hub.Module(decoder_spec, name="decoder_module")

        # Register and export encoder and decoder modules
        hub.register_module_for_export(encoder, "encoder")
        hub.register_module_for_export(decoder, "decoder")

        code = encoder(x, as_dict=True)
        recon = decoder(code['sample'])
        predictions = {'code': code['sample'], 'reconstruction': recon,
                       'log_prob':code['log_prob'], 'input':x}
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    with tf.variable_scope("encoder_module") as sc:
        encoding = encoder_model(x)
        code = encoding.sample()
        log_prob = encoding.log_prob(code)

    with tf.variable_scope("decoder_module") as sc:
        decoder_output = decoder_model(code)

    if tf.contrib.framework.is_tensor(decoder_output):
        recon = decoder_output
        loglikelihood = params['loglikelihood_fn'](x, recon, features)
    else:
        # In this case, the decoder is actually returning a distribution
        # which we can use to sample from and estimate the lihelihood function
        recon = decoder_output.sample()
        loglikelihood = decoder_output.log_prob(x)

    image_tile_summary("image", tf.to_float(x[:16]), rows=4, cols=4)
    if 'psf' in features.keys():
        r = tf.expand_dims(tf.spectral.irfft2d(tf.spectral.rfft2d(recon[:,:,:,0])*features['psf']),axis=-1)
    else:
        r = recon
    image_tile_summary("recon", tf.to_float(r[:16]), rows=4, cols=4)
    image_tile_summary("diff", tf.to_float(x[:16] - r[:16]), rows=4, cols=4)

    tf.summary.scalar('loglikelihood', tf.reduce_mean(loglikelihood))

    kl = log_prob - prior.log_prob(code)
    tf.summary.scalar('kl', tf.reduce_mean(kl))

    elbo = loglikelihood - kl*params['kl_weight']

    loss = - tf.reduce_mean(elbo)
    tf.summary.scalar("elbo", tf.reduce_mean(elbo))

    # Training of the model
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.cosine_decay(params["learning_rate"], global_step,
                                          params["max_steps"])
    tf.summary.scalar("learning_rate", learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=params['adam_epsilon'])
    grads_and_vars = optimizer.compute_gradients(loss)
    clipped_grads_and_vars = [(tf.clip_by_norm(grad, params["gradient_clipping"]), var) for grad, var in grads_and_vars]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=global_step)

    eval_metric_ops = {
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
                 kl_weight=0.001,
                 gradient_clipping=100,
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
        params['iaf_size'] = iaf_size
        params['kl_weight'] = kl_weight
        params['learning_rate'] = learning_rate
        params['max_steps'] = max_steps
        params['gradient_clipping'] = gradient_clipping

        super(self.__class__, self).__init__(model_fn=vae_model_fn,
                                             model_dir=model_dir,
                                             params=params,
                                             config=config)
