import timeit

import numpy as np
import tensorflow as tf

from libs.utils import save_images, mkdir
from net import DCGANGenerator, SNDCGAN_Discrminator
import _pickle as pickle
import sys
sys.path.append('../')

import galsim
cat = galsim.COSMOSCatalog(dir='/home/macaca/research/deep_galaxy_models/COSMOS_25.2_training_sample',
                                   file_name='real_galaxy_catalog_25.2.fits')

from deepgal.galsim import get_postage_stamp_map
from multiprocessing import Pool
import pdb

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_integer('max_iter', 100000, '')
flags.DEFINE_integer('snapshot_interval', 1000, 'interval of snapshot')
flags.DEFINE_integer('evaluation_interval', 10000, 'interval of evalution')
flags.DEFINE_integer('display_interval', 100, 'interval of displaying log to console')
flags.DEFINE_float('adam_alpha', 0.0001, 'learning rate')
flags.DEFINE_float('adam_beta1', 0.5, 'beta1 in Adam')
flags.DEFINE_float('adam_beta2', 0.999, 'beta2 in Adam')
flags.DEFINE_integer('n_dis', 1, 'n discrminator train')

mkdir('tmp')

#config = FLAGS.__flags
config = FLAGS.flag_values_dict()
generator = DCGANGenerator(**config)
discriminator = SNDCGAN_Discrminator(**config)

#dataset
pool=Pool()
dset = tf.data.Dataset.from_tensor_slices(cat.orig_index).batch(128).map(get_postage_stamp_map(cat.real_cat,stamp_size=32, pixel_size=0.06, pool=pool))
dset = dset.flat_map(lambda arg, *rest: tf.data.Dataset.from_tensor_slices((arg,) + rest))
dset = dset.repeat(2).cache('./cosmos_cache/cache32')
dset = dset.repeat().shuffle(buffer_size=2000).batch(config['batch_size']).prefetch(16)
iterator = dset.make_one_shot_iterator()
batch_im, batch_psf, batch_ps = iterator.get_next()
x = tf.reshape(batch_im,(config['batch_size'],32,32,1))

global_step = tf.Variable(0, name="global_step", trainable=False)
increase_global_step = global_step.assign(global_step + 1)
is_training = tf.placeholder(tf.bool, shape=())
z = tf.placeholder(tf.float32, shape=[None, generator.generate_noise().shape[1]])
x_hat = generator(z, is_training=is_training)
#add noise
x_noise = tf.spectral.rfft2d(tf.squeeze(x_hat)) * batch_psf
_noise = tf.squeeze(tf.random_normal(x_hat.shape))
_noise = tf.spectral.rfft2d(_noise) * tf.complex(tf.sqrt(tf.exp(batch_ps)),0*tf.sqrt(tf.exp(batch_ps)))
x_noise = tf.expand_dims(tf.spectral.irfft2d(x_noise + _noise),axis=-1)
#x = tf.placeholder(tf.float32, shape=x_hat.shape)

d_fake = discriminator(x_noise, update_collection=None)
# Don't need to collect on the second call, put NO_OPS
d_real = discriminator(x, update_collection="NO_OPS")
# Softplus at the end as in the official code of author at chainer-gan-lib github repository
d_loss = tf.reduce_mean(tf.nn.softplus(d_fake) + tf.nn.softplus(-d_real))
g_loss = tf.reduce_mean(tf.nn.softplus(-d_fake))
d_loss_summary_op = tf.summary.scalar('d_loss', d_loss)
g_loss_summary_op = tf.summary.scalar('g_loss', g_loss)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('snapshots')

d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.adam_alpha, beta1=FLAGS.adam_beta1, beta2=FLAGS.adam_beta2)
d_gvs = optimizer.compute_gradients(d_loss, var_list=d_vars)
g_gvs = optimizer.compute_gradients(g_loss, var_list=g_vars)
d_solver = optimizer.apply_gradients(d_gvs)
g_solver = optimizer.apply_gradients(g_gvs)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
if tf.train.latest_checkpoint('snapshots') is not None:
  saver.restore(sess, tf.train.latest_checkpoint('snapshots'))

np.random.seed(1337)
sample_noise = generator.generate_noise()
np.random.seed()
iteration = sess.run(global_step)
start = timeit.default_timer()

is_start_iteration = True
while iteration < FLAGS.max_iter:
  _, g_loss_curr = sess.run([g_solver, g_loss], feed_dict={z: generator.generate_noise(), is_training: True})
  for _ in range(FLAGS.n_dis):
    _, d_loss_curr, summaries = sess.run([d_solver, d_loss, merged_summary_op],
                                         feed_dict={z: generator.generate_noise(), is_training: True})
  # increase global step after updating G and D
  # before saving the model so that it will be written into the ckpt file
  sess.run(increase_global_step)
  if (iteration + 1) % FLAGS.display_interval == 0 and not is_start_iteration:
    summary_writer.add_summary(summaries, global_step=iteration)
    stop = timeit.default_timer()
    print('Iter {}: d_loss = {:4f}, g_loss = {:4f}, time = {:2f}s'.format(iteration, d_loss_curr, g_loss_curr, stop - start))
    start = stop
  if (iteration + 1) % FLAGS.snapshot_interval == 0 and not is_start_iteration:
    saver.save(sess, 'snapshots/model.ckpt', global_step=iteration)
    sample_images = sess.run(x_hat, feed_dict={z: sample_noise, is_training: False})
    save_images(sample_images, 'tmp/{:06d}.png'.format(iteration))
    sample_images = sess.run(x_noise, feed_dict={z: sample_noise, is_training: False})
    save_images(sample_images, 'tmp/n_{:06d}.png'.format(iteration))
  if (iteration + 1) % FLAGS.evaluation_interval == 0:
    sample_images = sess.run(x_hat, feed_dict={z: sample_noise, is_training: False})
    save_images(sample_images, 'tmp/{:06d}.png'.format(iteration))
    sample_images = sess.run(x_noise, feed_dict={z: sample_noise, is_training: False})
    save_images(sample_images, 'tmp/n_{:06d}.png'.format(iteration))
  iteration += 1
  is_start_iteration = False
