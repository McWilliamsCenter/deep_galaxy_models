from math import log2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import add_arg_scope

from .layers import wide_resnet

@add_arg_scope
def resnet_generator(code,
                     final_depth=16,
                     output_size=64,
                     output_channels=1,
                     output_activation_fn=None,
                     block_type='wide',
                     activation_fn=tf.nn.relu,
                     is_training=True,
                     keep_prob=None,
                     reuse=None,
                     outputs_collections=None, scope=None):
    """
    Defines a generator network

    Parameters
    ----------
    code: tensor
        Expects a rank 2 tensor containing the latent code
    block_type:
        Type of resnet block to use (default: wide)
    """
    if block_type  == 'wide':
        resnet_block = wide_resnet
    else:
        raise NotImplementedError

    # Computes the number of required stages to turn the input into an
    # image of the desired shape
    num_stages = int(log2(output_size))

    with tf.variable_scope(scope, [code], reuse=reuse) as sc:

        # First upscaling and reshaping into an image
        current_depth = final_depth * 2 ** (num_stages-2)
        net = tf.expand_dims(tf.expand_dims(net, 1),1)

        net = slim.conv2d_transpose(net, current_depth, kernel_size=4, stride=1,
                                     activation_fn=None, padding='VALID', scope='deconv1')

        #net = slim.dropout(net, keep_prob=0.9)

        # Now looping over stages
        for i in range(2, num_stages-1):
            current_depth = final_depth * 2 ** (num_stages - i - 1)
            net = resnet_block(net, current_depth, resample='up',keep_prob=keep_prob,
                               activation_fn=activation_fn, scope='resnet%d_a'%i)
            net = resnet_block(net, current_depth, resample=None,keep_prob=keep_prob,
                               activation_fn=activation_fn, scope='resnet%d_b'%i)

        # Upsampling last layer
        net = tf.image.resize_bilinear(net, [output_size,output_size], name='resize')
        # Final convolution into full resolution image
        output = slim.conv2d(net, output_channels,kernel_size=5, padding='SAME',
                             activation_fn=output_activation_fn, scope='conv_out' )

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)

@add_arg_scope
def resnet_encoder(inputs,
                   input_depth=16,
                   latent_size=64,
                   block_type='wide',
                   activation_fn=tf.nn.relu,
                   is_training=True,
                   reuse=None,
                   outputs_collections=None, scope=None):
    """Defines an encoder network based on resnet blocks

    """
    if block_type  == 'wide':
        resnet_block = wide_resnet
    else:
        raise NotImplementedError

    normalizer_fn = slim.batch_norm

    with tf.variable_scope(scope, [inputs], reuse=reuse) as sc:

        size_in = inputs.get_shape().as_list()[1]
        num_stages = int(log2(size_in))

        # Initial convolution
        net = slim.conv2d(inputs, input_depth, kernel_size=5, activation_fn=activation_fn, padding='SAME',
                          weights_initializer=slim.initializers.variance_scaling_initializer(),
                          normalizer_fn=None, stride=2, scope='conv_in')

        for  i in range(1, num_stages-2):
            current_depth = input_depth * 2**i
            net = resnet_block(net, current_depth, resample='down',
                              activation_fn=activation_fn, scope='resnet%d_a'%i)
            net = resnet_block(net, current_depth, resample=None,
                              activation_fn=activation_fn, scope='resnet%d_b'%i)

        # Reshaping into a 1D code
        net = slim.flatten(net, scope='flat')

        output = slim.fully_connected(net, 2*latent_size, activation_fn=None,
                                      normalizer_fn=None, scope='fc_enc1')

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)
