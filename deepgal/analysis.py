# This module contains a number of utility functions for analyising the generated
# images
import galsim
import numpy as np
from galsim.tensorflow.generative_model import GenerativeGalaxyModel
from astropy.table import Table, vstack
from multiprocessing import Pool
from functools import partial

def draw_galaxies(data_dir='/usr/local/share/galsim/COSMOS_25.2_training_sample',
                  generative_model='https://raw.githubusercontent.com/EiffL/GalSim-Hub/master/modules/generator.tar.gz',
                  batch_size=1024,
                  n_batches=None,
                  pool_size=12):
    """
    This function will draw in postage stamps a sample of galaxies using both
    the real COSMOS galaxy catalog and a given generative model, it outputs
    """

    cosmos_cat = galsim.COSMOSCatalog(dir=data_dir)
    cosmos_noise = galsim.getCOSMOSNoise()

    # Generating galaxies from the model by batch
    gal_model = GenerativeGalaxyModel(generative_model)
    sim_galaxies = []

    table_sims = None
    table_cosmos = None

    if n_batches is None:
        n_batches = len(cosmos_cat.orig_index )//batch_size

    # Process galaxies by batches
    for i in range(n_batches):
        inds = np.arange(i*batch_size,(i+1)*batch_size )
        print("Batch %d"%i)
        # Generate uncovolved light profiles
        sim_galaxies = gal_model.sample(cat=cosmos_cat.param_cat[cosmos_cat.orig_index[inds]])
        indices = [(j,k) for j,k in enumerate(inds)]

        # Draw galaxies on postage stamps
        engine = partial(_draw_galaxies, cosmos_cat=cosmos_cat,
                         sim_galaxies=sim_galaxies, cosmos_noise=cosmos_noise)
        if pool_size is None:
            res = []
            for ind in indices:
                res.append(engine(ind))
        else:
            with Pool(pool_size) as p:
                res = p.map(engine, indices)

        # Extract the postage stamps into separate lists, discarding the ones
        # that failed
        if table_sims is None:
            tmp_sims = []
            tmp_cosmos = []
        else:
            tmp_sims = [table_sims]
            tmp_cosmos = [table_cosmos]

        for k, im_sims, im_cosmos, moments_sims, moments_cosmos, flag in res:
            if flag:
                tmp_sims.append(moments_sims)
                tmp_cosmos.append(moments_cosmos)

        table_sims = vstack(tmp_sims)
        table_cosmos = vstack(tmp_cosmos)

        table_param = Table(cosmos_cat.param_cat[cosmos_cat.orig_index[:len(table_sims)]])

    return table_sims, table_cosmos, table_param

def _draw_galaxies(inds, cosmos_cat=None, cosmos_index=None, sim_galaxies=None, cosmos_noise=None):
    """ Function to draw the galaxies into postage stamps
    """
    i,k = inds
    im_sims = galsim.ImageF(64, 64, scale=0.03)
    im_cosmos = galsim.ImageF(64, 64, scale=0.03)
    flag=True

    try:
        cosmos_gal = cosmos_cat.makeGalaxy(k)
        psf = cosmos_gal.original_psf

        sims_gal = galsim.Convolve(sim_galaxies[i], psf)
        sims_gal.drawImage(im_sims,method='no_pixel')
        im_sims.addNoise(cosmos_noise)
        moments_sims = _moments({k: im_sims})

        cosmos_gal = galsim.Convolve(cosmos_gal, psf)
        cosmos_gal.drawImage(im_cosmos, method='no_pixel')
        moments_cosmos = _moments({k: im_cosmos})
    except:
        flag=False
        moments_sims=None
        moments_cosmos=None

    return (k, im_sims, im_cosmos, moments_sims, moments_cosmos, flag)


def _moments(images):
    """
    Computes HSM moments for a set of galsim images
    """
    sigma = []
    e  = []
    e1 = []
    e2 = []
    g  = []
    g1 = []
    g2 = []
    flag = []
    amp = []
    keys = []

    for k in images:
        shape = images[k].FindAdaptiveMom(guess_centroid=galsim.PositionD(32,32), strict=False)
        keys.append(k)
        amp.append(shape.moments_amp)
        sigma.append(shape.moments_sigma)
        e.append(shape.observed_shape.e)
        e1.append(shape.observed_shape.e1)
        e2.append(shape.observed_shape.e2)
        g.append(shape.observed_shape.g)
        g1.append(shape.observed_shape.g1)
        g2.append(shape.observed_shape.g2)
        if shape.error_message is not '':
            flag.append(False)
        else:
            flag.append(True)

    return Table({'index': keys,
                  'amp': amp,
                  'sigma_e': sigma,
                  'e': e,
                  'e1': e1,
                  'e2': e2,
                  'g': g,
                  'g1': g1,
                  'g2': g2,
                  'flag': flag})
