from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_hub as hub
tfd = tfp.distributions
tfb = tfp.bijectors

__all__ = ['FlowEstimator']

# def make_flow_spec(flow_fn, cond_size, latent_size):
#     # flow_fn is a trainable bijector
#     # cond_size is the size of the conditional tensor on which to condition the
#     def flow_module_fn():
#         cond_layer = tf.placeholder(tf.float32, shape=[None, cond_size])
#
#         # Creates the bijector and flow
#         with tf.variable_scope("flow", use_resource=False):
#             flow = flow_fn(cond_layer)
#
#         input_layer = tf.placeholder(tf.float32, shape=[None, latent_size])
#         hub.add_signature(inputs={'condition': cond_layer, 'x': input_layer},
#                           outputs=flow.log_prob(input_layer), name="log_prob")
#
#         hub.add_signature(inputs=cond_layer,
#                           outputs=flow.sample(tf.shape(cond_layer)[0]), name="sample")
#
#     return hub.create_module_spec(flow_module_fn)


def flow_model_fn(features, labels, mode, params, config):
    """
    Model function to create a VAE estimator
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    y = features['y']
    flow = params['flow_fn'](y)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = flow.sample(tf.shape(y)[0])
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    # Extract input images
    x = features['x']

    # Loads an encoding function to work on the images
    encoder = hub.Module(params['encoder_module'], trainable=False)
    code = encoder({'image': x, 'sample_shape': 1}, as_dict=True)

    # This is the loglikelihood of a batch of images
    loglikelihood = flow.log_prob(tf.reshape(code['sample'], (-1, code['sample'].shape[-1])))
    tf.summary.scalar('loglikelihood', tf.reduce_mean(loglikelihood))
    loss = - tf.reduce_mean(loglikelihood)

    # Training of the model
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.cosine_decay(params["learning_rate"],
                                          global_step,
                                          params["max_steps"])

    tf.summary.scalar("learning_rate", learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)

    eval_metric_ops = {"loglikelihood": tf.metrics.mean(tf.reduce_mean(loglikelihood))}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


class FlowEstimator(tf.estimator.Estimator):
    """An estimator for Vanilla Variational Auto-Encoders (VAEs).
    """

    def __init__(self,
                 flow_fn=None,
                 encoder_module=None,
                 sample_shape=1,
                 learning_rate=0.001,
                 max_steps=5001,
                 model_dir=None, config=None):
        """
        Args:
            encoder_fn: model function for the encoder
        """
        params = {}
        params['flow_fn'] = flow_fn
        params['encoder_module'] = encoder_module
        params['sample_shape'] = sample_shape
        params['learning_rate'] = learning_rate
        params['max_steps'] = max_steps

        super(self.__class__, self).__init__(model_fn=flow_model_fn,
                                             model_dir=model_dir,
                                             params=params,
                                             config=config)
