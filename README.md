# Deep Galaxy Models for GalSim

This repository hosts some code used to train deep generative models for use with
the GalSim software.

## Prerequisite

  - GalSim: See here to install it: https://github.com/GalSim-developers/GalSim
  - Tensorflow
  - Tensorflow Probability `pip install --user tensorflow-probability`
  - Tensorflow Hub `pip install --user tensorflow-hub`

The training data based on the COSMOS catalog is also required, see how to
download it here:


## To train a model

### VAE with conditional flow sampling

Training this model is a 2 step process, first train a normal variational
autoencoder, then train a conditional flow model to sample in the latent space.

```
$ python train_vae.py --model_dir=models/vae --export_dir=modules/vae
```
Then use the exported encoder and decoder modules to train a conditional sampling
model using the desired properties, for instance magnitude, size and redshift
```
$ python train_conditional_flow.py --conditions=mag_auto,flux_radius,zphot \
                                   --vae_modules=modules/vae \
                                   --model_dir=/data2/deepgal/FourierBasedFlow
```


use the `--help` option to see different training options


## Demos

  - [TestVAE.ipynb](notebooks/TestVAE.ipynb): Simple example of how to use the provided code to make a tf.Dataset from a GalSim galaxy catalog, and use it to train a simple VAE
  - [TestGAN.ipynb](notebooks/TestGAN.ipynb): Simple example of how to use tfgan to build a generative model.
