# ICLabel-Python

[![Codecov](https://codecov.io/gh/jacobf18/iclabel-python/branch/main/graph/badge.svg)](https://codecov.io/gh/jacobf18/iclabel-python)
[![GitHub Actions](https://github.com/jacobf18/iclabel-python/workflows/build/badge.svg)](https://github.com/jacobf18/iclabel-python/actions)
[![CircleCI](https://circleci.com/gh/jacobf18/iclabel-python.svg?style=shield)](https://circleci.com/gh/jacobf18/iclabel-python)
[![PyPI Download count](https://pepy.tech/badge/mne-icalabel)](https://pepy.tech/project/mne-icalabel)
[![Latest PyPI release](https://img.shields.io/pypi/v/mne-icalabel.svg)](https://pypi.org/project/mne-icalabel/)

This repository is a conversion of the popular ICLabel classifier for Python. In addition, we provide improvements in the form of other models.

## Ports Completed

1. Convert feature extraction
2. Convert Matlab ConvNet

## Ports TODO

1. Connect to mne package

## Converted Features

1. Autocorrelation
2. Power Spectral Density
3. Topomap

## Converting MatConvNet to PyTorch

Architecture in matconvnet:

<img src="ICLabel_DagNN_Architecture.png" width="400"/>

The PyTorch model is in the `PortToPytorch.ipynb` jupyter notebook. The state dict is in iclabelNet.pt.
The model has three inputs: image, psd, and autocorrelation features. To encourage generalization, the image
features are rotated and negated to quadruple the image features. The psd and autocorrelation features
are coppied to the new image features. Then, the predicted probabilities are averaged over all four
images.

## MNE Package Port


# Installation

To get the latest code using [git](https://git-scm.com/), open a terminal and type:

    git clone git://github.com/jacobf18/iclabel-python.git
    cd iclabel-python
    pip install -e .    

or one can install directly using pip

    pip install --user -U https://api.github.com/repos/jacobf18/iclabel-python/zipball/main

Alternatively, you can also download a
`zip file of the latest development version <https://github.com/mne-tools/mne-connectivity/archive/main.zip>`__.
