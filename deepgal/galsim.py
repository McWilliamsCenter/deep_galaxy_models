import tensorflow as tf
from multiprocessing import Pool
import numpy as np
import galsim
from absl import flags

FLAGS = flags.FLAGS

# Input pipeline parameters
flags.DEFINE_string("data_dir", default='/usr/local/share/galsim/COSMOS_25.2_training_sample',
                    help="Directory to the GalSim COSMOS data")

flags.DEFINE_string("filename", default='real_galaxy_catalog_25.2.fits',
                    help="Name of the COSMOS dataset")

flags.DEFINE_integer("stamp_size", default=64,
                    help="Size of the postage stamps")

flags.DEFINE_float("pixel_size", default=0.03,
                    help="Pixel size in arcsec")

flags.DEFINE_float("clip", default=1.,
                    help="Clip pixels by this value")

flags.DEFINE_integer("input_nprocs", default=12,
                    help="Number of parallel threads for the input pipeline")

flags.DEFINE_integer("nrepeat", default=10,
                    help="Number of times the dataset is augmented by rotations")

flags.DEFINE_string("cache_dir", default='/data2/COSMOS/cache64_10',
                    help="Path to directory storing a cache of the training set")

flags.DEFINE_list("conditions", default=[],
                    help="List of catalog fields to extract")


def build_input_pipeline(data_dir, filename='real_galaxy_catalog_25.2.fits',
                         conditions=[],
                         batch_size=128, stamp_size=64, pixel_size=0.03, clip=1.,
                         input_nprocs=None, nrepeat=10, cache_dir=None,
                         buffer_size=20000, **kwargs):
    """
    This function creates an input pipeline by drawing images from GalSim

    Parameters
    ----------
    dir: Directory for the GalSim data
    filename: Name of the GalSim real catalog
    conditions: List of catalog quantities to use as conditions
    nrepeat: Number of times the dataset is randomly rotated
    """
    cat = galsim.COSMOSCatalog(dir=data_dir, file_name=filename)

    # Extracts the parametric catalog
    cat_param = cat.param_cat[cat.orig_index]
    from numpy.lib.recfunctions import append_fields

    # Adds a few extra fields
    bparams = cat_param['bulgefit']
    sparams = cat_param['sersicfit']
    cat_param = append_fields(cat_param, 'bulge_q', bparams[:,11])
    cat_param = append_fields(cat_param, 'bulge_beta', bparams[:,15])
    cat_param = append_fields(cat_param, 'disk_q', bparams[:,3])
    cat_param = append_fields(cat_param, 'disk_beta', bparams[:,7])
    cat_param = append_fields(cat_param, 'bulge_hlr', cat_param['hlr'][:,1])
    cat_param = append_fields(cat_param, 'bulge_flux_log10', np.where(cat_param['use_bulgefit'] ==1, np.log10(cat_param['flux'][:,1]), np.zeros(len(cat_param) )))
    cat_param = append_fields(cat_param, 'disk_hlr', cat_param['hlr'][:,2])
    cat_param = append_fields(cat_param, 'disk_flux_log10', np.where(cat_param['use_bulgefit'] ==1, np.log10(cat_param['flux'][:,2]), np.zeros(len(cat_param) )))

    if input_nprocs is not None:
        pool = Pool(input_nprocs)
    else:
        pool = None

    def training_fn():
        dset = tf.data.Dataset.from_tensor_slices(cat.orig_index)
        dset = dset.batch(128).map(get_postage_stamp_map(cat.real_cat,
                                                         stamp_size=stamp_size,
                                                         pixel_size=pixel_size,
                                                         pool=pool))
        dset = dset.flat_map(lambda arg, *rest: tf.data.Dataset.from_tensor_slices((arg,) + rest))
        dset = dset.repeat(nrepeat)
        if cache_dir is not None:
            dset = dset.cache(cache_dir)
        if len(conditions) > 0:
            # Extract from the catalog the desired quantities
            dset_cond =tf.data.Dataset.zip(tuple([tf.data.Dataset.from_tensor_slices(cat_param[k].astype('float32')) for k in conditions]))
            dset_cond.repeat(nrepeat)
            dset = dset = tf.data.Dataset.zip((dset, dset_cond))
        dset = dset.repeat().shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(16)
        iterator = dset.make_one_shot_iterator()
        if len(conditions) == 0:
            batch_im, batch_psf, batch_ps = iterator.get_next()
            return {'x': tf.clip_by_value(batch_im, -clip, clip),
                    'psf':batch_psf,
                    'ps':batch_ps}, tf.clip_by_value(batch_im,-clip,clip)
        else:
            (batch_im, batch_psf, batch_ps), batch_cond = iterator.get_next()
            return {'x': tf.clip_by_value(batch_im,-clip,clip),
                    'psf':batch_psf,
                    'ps':batch_ps,
                    'y': {k:batch_cond[i] for i,k in enumerate(conditions)}}, tf.clip_by_value(batch_im,-clip,clip)
    return training_fn


def get_postage_stamp_map(real_galaxy_catalog, stamp_size=64, pixel_size=0.03, pool=None):
    """
    This function creates a mapping function from a batched dataset of galaxy
    indices to a dataset of postage stamps.

    Parameters
    ----------
    real_galaxy_catalog: RealGalaxyCatalog
        GalSim catalog instance

    stamp_size: int
        Size of the postage stamps to draw (default: 64)

    pixel_size: float
        Scale of the pixels in the postage stamp (default: 0.03)

    pool: multiprocessing.Pool (optional)
        Pool instance to use for parallel processing of the postage stamps

    Returns
    -------
    map_func: tf.py_func
        A tensorflow mapping function mapping galaxy indices to postage stamps
    """
    def _loading_function(index):
        gal_image = real_galaxy_catalog.getGalImage(index)
        psf_image = real_galaxy_catalog.getPSFImage(index)
        noise_image, noise_pix_scale, var = real_galaxy_catalog.getNoiseProperties(index)
        return (gal_image.array, psf_image.array, noise_image.array, noise_pix_scale, var, stamp_size, pixel_size)

    def _processing(index):
        params = map(_loading_function, index)
        if pool is None:
            res = list(map(_make_galaxy, params))
        else:
            res = pool.map(_make_galaxy, params)
        ims = np.stack([elem[0] for elem in res])
        psf = np.stack([elem[1] for elem in res])
        pss = np.stack([elem[2] for elem in res])
        return ims, psf, pss

    def func(x):
        im, psf ,ps =  tf.py_func(_processing, [x],
                                    [tf.float32, tf.complex64, tf.float32])
        im.set_shape([None, stamp_size, stamp_size])
        im = tf.clip_by_value(tf.expand_dims(im, axis=-1),-1,1)
        psf.set_shape([None, stamp_size, stamp_size // 2 + 1])
        ps.set_shape([None, stamp_size, stamp_size // 2 + 1])
        return im, psf, ps

    return func

def _make_galaxy(params):
    """
    Draws the galaxy, psf and noise power spectrum on a postage stamp
    """
    gal, psf, noise_image, in_pixel_scale, var, stamp_size, pixel_scale = params

    real_params = (galsim.Image(np.ascontiguousarray(gal), scale=in_pixel_scale),
                   galsim.Image(np.ascontiguousarray(
                       psf), scale=in_pixel_scale),
                   galsim.Image(np.ascontiguousarray(
                       noise_image), scale=in_pixel_scale),
                   in_pixel_scale, var)

    gal = galsim.RealGalaxy(real_params,
                            noise_pad_size=stamp_size * pixel_scale)

    psf = gal.original_psf
    gal = galsim.Convolve(gal, gal.original_psf)

    # Random rotation of the galaxy
    rotation_angle = galsim.Angle(-np.random.rand()
                                  * 2 * np.pi, galsim.radians)
    g = gal.rotate(rotation_angle)
    p = psf.rotate(rotation_angle)

    # Draw the Fourier domain image of the galaxy
    imC = galsim.ImageCF(stamp_size, stamp_size, scale=2. *
                        np.pi / (pixel_scale * stamp_size))
    imCp = galsim.ImageCF(stamp_size, stamp_size, scale=2. *
                         np.pi / (pixel_scale * stamp_size))
    g.drawKImage(image=imC)
    p.drawKImage(image=imCp)

    # Keep track of the pixels with 0 value
    mask = ~(np.fft.fftshift(imC.array)[:, :(stamp_size) // 2 + 1] == 0)

    # Inverse Fourier transform of the image
    # TODO: figure out why we need 2 fftshifts....
    im = np.fft.fftshift(np.fft.ifft2(
        np.fft.fftshift(imC.array))).real.astype('float32')

    # Transform the psf array into proper format for Theano
    im_psf = np.fft.fftshift(imCp.array)[:, :(
        stamp_size // 2 + 1)].astype('complex64')

    # Compute noise power spectrum
    ps = g.noise._get_update_rootps((stamp_size, stamp_size),
                                    wcs=galsim.PixelScale(pixel_scale))

    # The following comes from correlatednoise.py
    rt2 = np.sqrt(2.)
    shape = (stamp_size, stamp_size)
    ps[0, 0] = rt2 * ps[0, 0]
    # Then make the changes necessary for even sized arrays
    if shape[1] % 2 == 0:  # x dimension even
        ps[0, shape[1] // 2] = rt2 * ps[0, shape[1] // 2]
    if shape[0] % 2 == 0:  # y dimension even
        ps[shape[0] // 2, 0] = rt2 * ps[shape[0] // 2, 0]
        # Both dimensions even
        if shape[1] % 2 == 0:
            ps[shape[0] // 2, shape[1] // 2] = rt2 * \
                ps[shape[0] // 2, shape[1] // 2]

    # Apply mask to power spectrum so that it is very large outside maxk
    ps = np.where(mask, np.log(ps**2), 10).astype('float32')

    return im, im_psf, ps
