###
API
###

.. automodule:: mne_icalabel
   :no-members:
   :no-inherited-members:

This is the application programming interface (API) reference
for classes (``CamelCase`` names) and functions
(``underscore_case`` names) of MNE-ICALabel.

Most-used functions
===================

.. currentmodule:: mne_icalabel

.. autosummary::
   :toctree: generated/

   label_components

ICLabel
=======

This is the model originally available for `EEGLab <https://github.com/sccn/ICLabel>`_.
The model was ported from matconvnet using `pytorch <https://pytorch.org/>`_.

Architecture:

.. image:: _static/ICLabel_DagNN_Architecture.png
   :width: 400
   :alt: ICLabel Neural Network Architecture

The model has three inputs: image, psd, and autocorrelation features. To encourage generalization, the image
features are rotated and negated, thus quadrupling the feature. The psd and autocorrelation features
are copied to the new image features. Then, the predicted probabilities are averaged over all four images.

.. currentmodule:: mne_icalabel.iclabel

.. autosummary::
   :toctree: generated/

   get_iclabel_features
   run_iclabel

Features
========

Contains functions to extract features from `~mne.preprocessing.ICA` instance and `~mne.io.Raw` or
`~mne.Epochs` instances using MNE-Python.

.. currentmodule:: mne_icalabel.features

.. autosummary::
   :toctree: generated/

   get_topomap
   get_topomaps

Annotating Components
=====================
To facilitate annotation of the ICA components, we provide an API that conforms to the
derivative standard of BIDS for EEG data.

.. currentmodule:: mne_icalabel.annotation

.. autosummary::
   :toctree: generated/

   mark_component
   write_components_tsv
