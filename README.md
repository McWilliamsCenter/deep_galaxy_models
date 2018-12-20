# Deep Galaxy Models for GalSim

This repository hosts some code used to train deep generative models for use with
the GalSim software.

## Prerequisite

  - GalSim: See here to install it: https://github.com/GalSim-developers/GalSim
  - Tensorflow
  - Tensorflow Probability
  - Tensorflow Hub

## To train a model

Simply use the provided training function:

```
$ python train_vae.py --model_dir=models/vae
```
use the `--help` option to see different training options

## Demos

  - [TestVAE.ipynb](notebooks/TestVAE.ipynb): Simple example of how to use the provided code to make a tf.Dataset from a GalSim galaxy catalog, and use it to train a simple VAE
  - [TestGAN.ipynb](notebooks/TestGAN.ipynb): Simple example of how to use tfgan to build a generative model.
