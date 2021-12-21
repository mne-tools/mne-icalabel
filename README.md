# ICLabel-Python

This repository is a conversion of the popular ICLabel classifier for python.

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
