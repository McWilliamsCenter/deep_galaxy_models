import galsim
import numpy as np
from galsim_hub import GenerativeGalaxyModel
from multiprocessing import Pool
from functools import partial
from astropy.table import Table, vstack, join
from .stats import moments, morph_stats

STAMP_SIZE=128
PIXEL_SCALE=0.03

def draw_galaxies(data_dir='/usr/local/share/galsim/COSMOS_25.2_training_sample',
                  generative_model='https://raw.githubusercontent.com/EiffL/GalSim-Hub/master/modules/generative_model.tar.gz',
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
    sims_stamps = []
    cosmos_stamps = []
    param_stamps = []
    idents = []

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
        for k, im_sims, im_cosmos, im_param, flag in res:
            if flag:
                sims_stamps.append(im_sims)
                cosmos_stamps.append(im_cosmos)
                param_stamps.append(im_param)
                idents.append(cosmos_cat.param_cat[cosmos_cat.orig_index[k]]['IDENT'])

    # Puts all images into one big table
    table = Table([np.array(idents),
                   np.stack(cosmos_stamps),
                   np.stack(sims_stamps),
                   np.stack(param_stamps)], names=['IDENT', 'real', 'mock', 'param'])

    # Merge with the cosmos catalog data for convenience
    table = join(cosmos_cat.param_cat, table, keys=['IDENT'])

    return table

def _draw_galaxies(inds, cosmos_cat=None, cosmos_index=None, sim_galaxies=None, cosmos_noise=None):
    """ Function to draw the galaxies into postage stamps
    """
    i,k = inds
    im_sims = galsim.ImageF(STAMP_SIZE, STAMP_SIZE, scale=PIXEL_SCALE)
    im_param = galsim.ImageF(STAMP_SIZE, STAMP_SIZE, scale=PIXEL_SCALE)
    im_cosmos = galsim.ImageF(STAMP_SIZE, STAMP_SIZE, scale=PIXEL_SCALE)
    flag=True
    big_fft_params = galsim.GSParams(maximum_fft_size=50000)

    try:
        cosmos_gal = cosmos_cat.makeGalaxy(k)
        param_gal = cosmos_cat.makeGalaxy(k, gal_type='parametric')
        psf = cosmos_gal.original_psf

        sims_gal = galsim.Convolve(sim_galaxies[i], psf)
        sims_gal.drawImage(im_sims,method='no_pixel')
        im_sims.addNoise(cosmos_noise)

        cosmos_gal = galsim.Convolve(cosmos_gal, psf)
        cosmos_gal.drawImage(im_cosmos, method='no_pixel')

        param_gal = galsim.Convolve(param_gal, psf)
        param_gal.drawImage(im_param, method='no_pixel')
        im_param.addNoise(cosmos_noise)
    except:
        flag=False

    return (k, im_sims.array, im_cosmos.array, im_param.array, flag)

def compute_statistics(postage_stamps, pool_size=12):
    """
    Copmutes a set of statistics on the provided images, using
    """

    with Pool(pool_size) as p:
        tables = []
        for t in ['real', 'mock', 'param']:
            hsm_table = vstack(p.map(moments, postage_stamps[t]))
            hsm_table['IDENT'] = postage_stamps['IDENT']
            #limiting the R code to images 64x64
            stats_table = vstack(p.map(morph_stats, postage_stamps[t][:,32:-32,32:-32]))
            stats_table['IDENT'] = postage_stamps['IDENT']
            table = join(hsm_table, stats_table, keys=['IDENT'], table_names=['moments', 'morph'])
            table['flag'] = table['flag_moments'] & table['flag_morph']
            tables.append(table)

    return tables[0], tables[1], tables[2]
