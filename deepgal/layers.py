import tensorflow as tf
import tensorflow.contrib.slim as slim


@slim.add_arg_scope
def wide_resnet(inputs, depth, resample=None,
                keep_prob=None,
                activation_fn=tf.nn.relu,
                is_training=True,
                outputs_collections=None, scope=None):
    """
    Wide residual units as advocated in arXiv:1605.07146
    Adapted from slim implementation of residual networks
    Resample can be 'up', 'down', 'none'
    """
    depth_residual = 2*depth
    # if resample is None:
    #     stride = 1
    # else:
    #     stride = 2
    #
    stride = 1
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    size_in = inputs.get_shape().as_list()[1]

    with tf.variable_scope(scope, 'wide_resnet', [inputs]) as sc:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d_transpose, slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                activation_fn=activation_fn,
                                weights_initializer=slim.initializers.variance_scaling_initializer(),
                                kernel_size=3,
                                stride=1):

                preact = slim.batch_norm(inputs, activation_fn=activation_fn, scope='preact')

                if resample == 'up':
                    output_size = size_in*2
                    # Apply bilinear upsampling
                    preact = tf.image.resize_bilinear(preact, [output_size, output_size], name='resize')
                elif resample == 'down':
                    output_size = size_in/2
                    preact = slim.avg_pool2d(preact, kernel_size=[2,2], stride=2, padding='SAME', scope='resize')


                if depth_in != depth:
                    shortcut = slim.conv2d(preact, depth, kernel_size=1, normalizer_fn=None, activation_fn=None, scope='shortcut')
                else:
                    shortcut = preact

                residual = slim.conv2d(preact, depth_residual, scope='res1')

                if keep_prob is not None:
                    residual = slim.dropout(residual, keep_prob=keep_prob)

                residual = slim.conv2d(residual, depth, stride=1, scope='res2',
                                        normalizer_fn=None, activation_fn=None)

                output = shortcut + residual

                return slim.utils.collect_named_outputs(outputs_collections, sc.name,
                                                        output)
