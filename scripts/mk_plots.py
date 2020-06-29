import os
import sys
from astropy.table import Table
from deepgal.validation.utils import draw_galaxies, compute_statistics
from deepgal.validation.plotting import moments_plots
from absl import flags, app
import pathlib

flags.DEFINE_integer("n_batches", default=20,
                  help="Number of batches of galaxies to sample")

flags.DEFINE_integer("batch_size", default=512,
                  help="Size of batch of galaxies")

flags.DEFINE_integer("pool_size", default=12,
                  help="Number of processes to use in parallel")

flags.DEFINE_string("data_dir", default='/usr/local/share/galsim/COSMOS_25.2_training_sample',
                    help="Directory to the GalSim COSMOS data")

flags.DEFINE_string("out_dir", default="./results",
                    help="Path to directory where to save the plots and results")

flags.DEFINE_string("generative_model", default='modules/flow_vae_cosmos_128_realnvp/generator',
                    help="Generative model to use when sampling galaxies with GalSim")

FLAGS = flags.FLAGS


def main(argv):
    del argv  # unused

    pathlib.Path(FLAGS.out_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(FLAGS.out_dir+"/plots/").mkdir(parents=True, exist_ok=True)

    stamps_path = os.path.join(FLAGS.out_dir, 'postage_stamps.fits')
    cat_real_path = os.path.join(FLAGS.out_dir, 'catalog_real.fits')
    cat_mock_path = os.path.join(FLAGS.out_dir, 'catalog_mock.fits')
    cat_param_path = os.path.join(FLAGS.out_dir, 'catalog_param.fits')

    # First, draw the postage stamps if they don't already exist
    if os.path.isfile(stamps_path):
        postage_stamps = Table.read(stamps_path)
    else:
        postage_stamps = draw_galaxies(generative_model=FLAGS.generative_model,
                                       n_batches=FLAGS.n_batches,
                                       batch_size=FLAGS.batch_size,
                                       pool_size=FLAGS.pool_size)
        print("Saving postage stamps in ", stamps_path)
        postage_stamps.write(stamps_path)
    print("Done loading postage stamps")

    # Second, compute statistics if it's not already done
    if any([not os.path.isfile(p) for p in [cat_real_path, cat_mock_path, cat_param_path]]):
        m_real, m_mock, m_param = compute_statistics(postage_stamps,
                                                     pool_size=FLAGS.pool_size)
        print("Saving catalogs")
        m_real.write(cat_real_path)
        m_mock.write(cat_mock_path)
        m_param.write(cat_param_path)
    else:
        m_real = Table.read(cat_real_path)
        m_mock = Table.read(cat_mock_path)
        m_param = Table.read(cat_param_path)
    print("Done loading morphology statistics")

    moments_plots(postage_stamps, m_real, m_mock, prefix=FLAGS.out_dir+"/plots/")

if __name__ == "__main__":
    app.run(main)
