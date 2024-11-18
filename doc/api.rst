API
===

.. automodule:: mne_icalabel
   :no-members:
   :no-inherited-members:

This is the application programming interface (API) reference
for classes (``CamelCase`` names) and functions
(``underscore_case`` names) of MNE-ICALabel.

Most-used functions
-------------------

.. currentmodule:: mne_icalabel

.. autosummary::
   :toctree: ./generated/api

   label_components

ICLabel
-------

This is the model originally available for `EEGLab <https://github.com/sccn/ICLabel>`_.
The model was ported from matconvnet using `pytorch <https://pytorch.org/>`_ or
`Microsoft onnxruntime <https://onnxruntime.ai/>`_.

ICLabel is designed to classify ICs fitted with an extended infomax ICA
decomposition algorithm on EEG datasets referenced to a common average and
filtered between [1., 100.] Hz. It is possible to run ICLabel on datasets that
do not meet those specification, but the classification performance
might be negatively impacted. Moreover, the ICLabel paper did not study the
effects of these preprocessing steps.

Architecture:

.. image:: _static/ICLabel_DagNN_Architecture.png
   :width: 400
   :alt: ICLabel Neural Network Architecture
   :align: center

The model has three inputs: image (topomap), psd, and autocorrelation features.
To encourage generalization, the image feature is rotated and negated, thus
quadrupling the feature. After 3 convolutional layer with a ReLu activation,
the 3 features are concatenated for the final layer.

.. currentmodule:: mne_icalabel.iclabel

.. autosummary::
   :toctree: ./generated/api

   iclabel_label_components
   get_iclabel_features
   run_iclabel

Features
--------

Contains functions to extract features from `~mne.preprocessing.ICA` instance and `~mne.io.Raw` or
`~mne.Epochs` instances using MNE-Python.

.. currentmodule:: mne_icalabel.features

.. autosummary::
   :toctree: ./generated/api

   get_topomaps

Annotating Components
---------------------

To facilitate annotation of the ICA components, we provide an API that conforms to the
derivative standard of BIDS for EEG data.

.. currentmodule:: mne_icalabel.annotation

.. autosummary::
   :toctree: ./generated/api

   mark_component
   write_components_tsv

In addition, as of v0.3, we have introduced a beta-version of a GUI that
assists in annotated ICA components. This was heavily inspired by the annotation
process in ``ICLabel``. If you use this feature, please note that there may be
significant bugs still. Please report these in the GH issues tab.

.. currentmodule:: mne_icalabel

.. autosummary::
   :toctree: ./generated/api

   gui.label_ica_components
