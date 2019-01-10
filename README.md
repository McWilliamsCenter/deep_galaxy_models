# Deep Galaxy Models for GalSim

This repository hosts some code used to train deep generative models for use with
the GalSim software.

## Prerequisite

  - GalSim: See here to install it: https://github.com/GalSim-developers/GalSim
  - GalSim-Hub: `pip install --user galsim-hub`
  - Tensorflow
  - Tensorflow Probability `pip install --user tensorflow-probability`
  - Tensorflow Hub `pip install --user tensorflow-hub`

The training data based on the COSMOS catalog is also required, see how to
download it here:


## To train a model

### VAE with conditional flow sampling

Training this model is a 2 step process, first train a normal variational
autoencoder, then train a conditional flow model to sample in the latent space.

```sh
$ python train_vae.py --model_dir=models/vae --export_dir=modules/vae
```
Then use the exported encoder and decoder modules to train a conditional sampling
model using the desired properties, for instance magnitude, size and redshift
```sh
$ python train_conditional_flow.py --conditions=mag_auto,flux_radius,zphot \
                                   --vae_modules=modules/vae \
                                   --model_dir=/data2/deepgal/FourierBasedFlow
```


use the `--help` option to see different training options

## Exporting a module for GalSim

GalSim will expect a module with a number of input quantities
such as `mag_auto` or `flux_radius`, and outputting an *unconvolved* light
profile on a postage stamp. In addition, the module should store as attached
messages the size of the postage stamp, and the pixel scale. All of this is
done by `train_conditional_flow.py` which saves a module named `generator` in
the requested `export_dir`.

To archive the module before making available online, do the following from inside the `generator` folder:
```sh
$ tar -cz -f ../generative_model.tar.gz --owner=0 --group=0 .
```
This will compress the module as a tar.gz archive, which can now be hosted online
and directly ingested by GalSim.

## Testing the generative model

The code for making some comparison plots is provided in the `deepgal.validation`
module. To automatically produce all the diagnostic plots run the following
command:

```sh
$ python mk_plots --generative_model=modules/flow/generator
```

This will use the generator specified to draw some postage stamps and then compute
some statistics on these before producing the plots.

## Old Demos
These are old examples

  - [TestVAE.ipynb](notebooks/TestVAE.ipynb): Simple example of how to use the provided code to make a tf.Dataset from a GalSim galaxy catalog, and use it to train a simple VAE
  - [TestGAN.ipynb](notebooks/TestGAN.ipynb): Simple example of how to use tfgan to build a generative model.
