# This file contains analysis routines used to compute various
# statistics on real and simulated images
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
from astropy.table import Table
import pandas as pd
import galsim
import os
import numpy as np

def moments(images, scale=0.03):
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
    images = np.atleast_3d(images)
    if images.shape[-1] == 1:
        images = np.swapaxes(images, 0, 2)
    for i in range(len(images)):
        image = galsim.Image(images[i], scale=scale)
        shape = image.FindAdaptiveMom(guess_centroid=galsim.PositionD(32,32), strict=False)
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

    return Table({'amp': amp,
                  'sigma_e': sigma,
                  'e': e,
                  'e1': e1,
                  'e2': e2,
                  'g': g,
                  'g1': g1,
                  'g2': g2,
                  'flag': flag})

def morph_stats(images):
    """
    Computes various morphology statistics provided by Peter Freeman
    """
    # Initialize the R interface and extract R function
    ro.r('source("%s/morphological_indicators/interface.R", chdir=T)'%os.path.dirname(__file__))
    compute_statistics_single = ro.r.compute_statistics_single
    pandas2ri.activate()
    numpy2ri.activate()

    flag = []
    rows = []

    images = np.atleast_3d(images)
    if images.shape[-1] == 1:
        images = np.swapaxes(images, 0, 2)

    for i in range(len(images)):
        im = images[i]
        ret = compute_statistics_single(im)
        # Testing if successful
        if(ret[0][0]):
            flag.append(True)
        else:
            flag.append(False)
        rows.append(pandas2ri.ri2py(ret[1]))

    # Convert pandas dataframe to astropy table
    tab = Table.from_pandas(pd.concat(rows))
    tab['flag'] = flag
    return tab
