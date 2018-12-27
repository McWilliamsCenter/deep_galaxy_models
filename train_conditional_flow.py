import os
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
from deepgal.galsim import build_input_pipeline
from deepgal.flow import masked_autoregressive_conditional_template
from deepgal.FlowEstimator import flow_model_fn

# Encoder parameter
flags.DEFINE_string("encoder_module", default=None,
                     help="Encoder module.")

# Model parameters
flags.DEFINE_integer("maf_layers", default=3,
                     help="Number of MAF layers")

flags.DEFINE_integer("maf_size", default=512,
                     help="Number of hidden neurons per MAF layers.")

flags.DEFINE_integer("latent_size", default=128,
                     help="Number of dimensions in the latent code (z).")

flags.DEFINE_string("activation", default="leaky_relu",
                     help="Activation function for all hidden layers.")

flags.DEFINE_bool("shift_only", default="False",
                     help="Whether to build a shift only MAF.")

# Training parameters
flags.DEFINE_integer("batch_size", default=256,
                     help="Batch size.")

flags.DEFINE_float("learning_rate", default=0.001,
                     help="Initial learning rate.")

flags.DEFINE_integer("max_steps", default=50001,
                     help="Number of training steps to run.")

flags.DEFINE_string("model_dir", default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "flow/"),
                     help="Directory to put the model's fit.")

flags.DEFINE_string("export_dir", default="modules/flow",
                     help="Directory to put the trained tensorflow modules.")

flags.DEFINE_integer("save_checkpoints_steps", default=1000,
                     help="Frequency at which to save checkpoints.")

FLAGS = flags.FLAGS

def make_flow_fn(latent_size, maf_layers, maf_size, shift_only, activation):
    """ Creates a flow function with provided parameters
    """
    def flow_fn(cond, is_training):
        def init_once(x, name, trainable=False):
            return tf.get_variable(name, initializer=x, trainable=trainable)

        # Apply batch normalization on the inputs
        cond = tf.layers.batch_normalization(cond, axis=-1, training=is_training)

        chain = []
        for i in range(maf_layers):
            if i < 2:
                chain.append(tfb.MaskedAutoregressiveFlow(
                            shift_and_log_scale_fn=masked_autoregressive_conditional_template(
                                hidden_layers=[maf_size, maf_size],
                                conditional_tensor=cond,
                                activation=activation,
                                shift_only=False, name='maf%d'%i)))
            else:
                chain.append(tfb.MaskedAutoregressiveFlow(
                            shift_and_log_scale_fn=masked_autoregressive_conditional_template(
                                hidden_layers=[maf_size, maf_size],
                                conditional_tensor=cond,
                                activation=activation,
                                shift_only=shift_only, name='maf%d'%i)))
            chain.append(tfb.Permute(permutation=init_once(
                                 np.random.permutation(latent_size).astype("int32"),
                                 name='permutation%d'%i)))
        chain = tfb.Chain(chain)

        flow = tfd.TransformedDistribution(
                distribution=tfd.MultivariateNormalDiag(loc=np.zeros(latent_size, dtype='float32'),
                                                        scale_diag=init_once(np.ones(latent_size, dtype='float32'),
                                                        name='latent_scale', trainable=(not shift_only))),
                bijector=chain)

        return flow

    return flow_fn

def main(argv):
    del argv  # unused

    params = FLAGS.flag_values_dict()
    params["activation"] = getattr(tf.nn, params["activation"])
    params["flow_fn"] = make_flow_fn(latent_size=params["latent_size"],
                                     maf_layers=params["maf_layers"],
                                     maf_size=params["maf_size"],
                                     shift_only=params["shift_only"],
                                     activation=params["activation"])

    tf.gfile.MakeDirs(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.cache_dir)

    # Build input pipeline
    input_fn = build_input_pipeline(**params)

    # Build estimator
    estimator = tf.estimator.Estimator(
      flow_model_fn,
      params=params,
      config=tf.estimator.RunConfig(
          model_dir=FLAGS.model_dir,
          save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      ))

    estimator.train(input_fn=input_fn, max_steps=FLAGS.max_steps)

    exporter = hub.LatestModuleExporter("tf_hub",
        tf.estimator.export.build_raw_serving_input_receiver_fn(input_fn()[0]['y']))
    exporter.export(estimator, FLAGS.export_dir, estimator.latest_checkpoint())

if __name__ == "__main__":
    tf.app.run()
