# Deep Generative Models for Galaxy Image Simulations

[![arXiv:2008.03833](https://img.shields.io/badge/astro--ph.IM-arXiv%3A2008.03833-B31B1B.svg)](https://arxiv.org/abs/2008.03833)

This repository hosts the analysis code for the `Deep Generative Models for Galaxy Image Simulations`
paper.

Content:
  - Notebooks for reproducing each figures
  - Scripts used to train the generative model and run the evaluation in `scripts`
  - All components of the generative model (AutoEncoder, Latent Flow,  VAE-Flow)
  as TF-Hub modules in `modules`.

The data used to make the plots of the paper is available here: https://zenodo.org/record/3975700

## How to use this repository

The first step in order to execute the notebooks is to download the data resulting
from the main analysis script (see below for more details). You can do so using the
following command:

```bash
$ wget -O results.tgz https://zenodo.org/record/3975700/files/results_lanusse2020.tar.gz?download=1
$ tar -xvzf results.tgz
```

This will download and extract the postage stamps of COSMOS images and of mock
galaxy images used in the paper. The archive also contains catalogs of morphological
statistics computed on each stamps.

With the data downloaded, the dependencies needed to run the notebooks are:
  - matplotlib
  - seaborn
  - astropy
  - GalSim: See here to install it: https://github.com/GalSim-developers/GalSim
  - TensorFlow version 1.15 (not compatible with TF 2.x)
  - TensorFlow-Hub : `pip install --user tensorflow-hub`
  - GalSim-Hub : `pip install --user galsim-hub`
  - daft: `pip install --user daft`

Then you should be able to run all the notebooks.
