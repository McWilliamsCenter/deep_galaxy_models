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


def flow_model_fn(features, labels, mode, params, config):
    """
    Model function to create a VAE estimator
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)


    if mode == tf.estimator.ModeKeys.PREDICT:
        y = features
        def flow_module_spec():
            inputs = {k: tf.placeholder(tf.float32, shape=[None]) for k in y.keys()}
            cond_layer = tf.concat([tf.expand_dims(inputs[k], axis=1) for k in inputs.keys()],axis=1)
            flow = params['flow_fn'](cond_layer, is_training)
            hub.add_signature(inputs=inputs,
                              outputs=flow.sample(tf.shape(cond_layer)[0]))

        flow_spec = hub.create_module_spec(flow_module_spec)
        flow = hub.Module(flow_spec, name='flow_module')
        hub.register_module_for_export(flow, "code_sampler")
        predictions = {'code': flow(y)}
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    x = features['x']
    y = features['y']

    # Loads the encoding function to work on the images
    encoder = hub.Module(params['encoder_module'], trainable=False)
    code = encoder(x, as_dict=True)

    with tf.variable_scope("flow_module"):
        cond_layer = tf.concat([tf.expand_dims(y[k], axis=1) for k in y.keys()],axis=1)
        flow = params['flow_fn'](cond_layer, is_training)
        loglikelihood = flow.log_prob(code['sample'])

    # This is the loglikelihood of a batch of images
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
                 learning_rate=0.001,
                 max_steps=50001,
                 model_dir=None, config=None):
        """
        Args:
            encoder_fn: model function for the encoder
        """
        params = {}
        params['flow_fn'] = flow_fn
        params['encoder_module'] = encoder_module
        params['learning_rate'] = learning_rate
        params['max_steps'] = max_steps

        super(self.__class__, self).__init__(model_fn=flow_model_fn,
                                             model_dir=model_dir,
                                             params=params,
                                             config=config)
