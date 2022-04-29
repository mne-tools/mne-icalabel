###
API
###

:py:mod:`mne_icalabel`:

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

.. currentmodule:: mne_icalabel.iclabel

.. autosummary::
   :toctree: generated/

This is the model originally available in `EEGLab <https://github.com/sccn/ICLabel>`_.
The model was ported using matconvnet.

Architecture in matconvnet:

   <img src="ICLabel_DagNN_Architecture.png" width="400"/>

The model has three inputs: image, psd, and autocorrelation features. To encourage generalization, the image
features are rotated and negated to quadruple the image features. The psd and autocorrelation features
are copied to the new image features. Then, the predicted probabilities are averaged over all four images.

   get_iclabel_features
   ICLabelNet
   run_iclabel