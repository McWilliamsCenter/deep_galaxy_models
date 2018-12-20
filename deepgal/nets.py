import tensorflow as tf
from .layers import wide_resnet

def resnet_decoder(code,
                     base_depth=128,
                     num_stages=1,
                     output_size=64,
                     output_channels=1,
                     activation=tf.nn.leaky_relu,
                     is_training=True, reuse=None, scope=None):
    """
    Defines a generator network

    Parameters
    ----------
    code: tensor
        Expects a rank 2 tensor containing the latent code
    """

    with tf.variable_scope(scope, [code], reuse=reuse) as sc:

        # First upscaling and reshaping into an image

        net = tf.reshape(code, (-1, 1, 1, code.shape[-1]))
        current_depth = base_depth * 2**num_stages
        net = tf.layers.conv2d_transpose(net, current_depth, kernel_size=4, strides=2,
                                         activation=activation, padding='VALID', name='deconv1')
        # Now looping over stages
        for i in range(1, num_stages+1):
            current_depth = base_depth * 2 ** (num_stages - i)
            net = wide_resnet(net, current_depth, resample='up', activation_fn=activation, scope='resnet%d_a'%i, is_training=is_training)
            net = wide_resnet(net, current_depth, resample=None, activation_fn=activation, scope='resnet%d_b'%i, is_training=is_training)

        net = tf.layers.conv2d_transpose(net, base_depth, kernel_size=4, strides=2,
                                         activation=activation, padding='SAME', name='deconv2')
        net = tf.layers.batch_normalization(net, training=is_training, name='bn1')
        net = tf.layers.conv2d_transpose(net, base_depth, kernel_size=4, strides=2,
                                         activation=activation, padding='SAME', name='deconv3')

        output = tf.layers.conv2d_transpose(net, output_channels, kernel_size=4, strides=2,
                                         activation=None, padding='SAME', name='deconv4')

        return output


def resnet_encoder(inputs,
                   base_depth=128,
                   num_stages=1,
                   latent_size=128,
                   activation=tf.nn.leaky_relu,
                   is_training=True, reuse=None, scope=None):
    """Defines an encoder network based on resnet blocks
    """
    with tf.variable_scope(scope, [inputs], reuse=reuse) as sc:

        net = tf.layers.conv2d(inputs, base_depth, kernel_size=4, activation=activation, strides=2, name='conv1')
        net = tf.layers.batch_normalization(net, training=is_training, name='bn1')
        net = tf.layers.conv2d(net, base_depth, kernel_size=4, activation=activation, strides=2, name='conv2')

        for i in range(1, num_stages+1):
            print(i)
            current_depth = base_depth * 2**i
            net = wide_resnet(net, current_depth, resample=None, scope='resnet%d_a'%i, is_training=is_training)
            net = wide_resnet(net, current_depth, resample='down', scope='resnet%d_b'%i, is_training=is_training)

        net = tf.layers.conv2d(net, latent_size, kernel_size=3, strides=2, activation=activation)

        # Reshaping into a 1D code
        net = tf.layers.flatten(net)
        output = tf.layers.dense(net, 2*latent_size, activation=None, name='fc_enc1')

        return output
