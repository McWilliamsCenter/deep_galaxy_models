# This scripts allows you to train a VAE and produce and encoding and
# decoding modules

import tensorflow as tf
import os
from absl import flags

flags.DEFINE_float("learning_rate", default=0.001,
                     help="Initial learning rate.")

flags.DEFINE_integer("max_steps", default=10001,
                     help="Number of training steps to run.")

flags.DEFINE_integer("latent_size", default=16,
                     help="Number of dimensions in the latent code (z).")


flags.DEFINE_integer("batch_size", default=32, help="Batch size.")


flags.DEFINE_integer("n_samples", default=16, help="Number of samples to use in encoding.")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/"),
    help="Directory to put the model's fit.")
