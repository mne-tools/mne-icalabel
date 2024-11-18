:orphan:

ICLabel
=======

This is the model originally available for `EEGLab <https://github.com/sccn/ICLabel>`_.
The model was ported from matconvnet using `pytorch <https://pytorch.org/>`_ or
`Microsoft onnxruntime <https://onnxruntime.ai/>`_.

ICLabel is designed to classify ICs fitted with an extended infomax ICA
decomposition algorithm on *EEG datasets* referenced to a common average and
filtered between (1, 100) Hz. It is possible to run ICLabel on datasets that
do not meet those specification, but the classification performance
might be negatively impacted. Moreover, the ICLabel paper did not study the
effects of these preprocessing steps.

Architecture
------------

.. image:: ../_static/ICLabel_DagNN_Architecture.png
    :width: 400
    :alt: ICLabel Neural Network Architecture
    :align: center

.. raw:: html

    <p style="margin-bottom: 15px;"></p>

The model has three inputs: image (topomap), psd, and autocorrelation features.
To encourage generalization, the image feature is rotated and negated, thus
quadrupling the feature. After 3 convolutional layer with a ReLu activation,
the 3 features are concatenated for the final layer.

API
---

.. currentmodule:: mne_icalabel.iclabel

.. autosummary::
   :toctree: ../generated/api

   iclabel_label_components
   get_iclabel_features
   run_iclabel

Cite
----

If you use ICLabel, please also cite the original
paper\ :footcite:p:`PionTonachini2019`.

.. footbibliography::

.. dropdown:: BibTeX for ICLabel
   :color: info

.. code-block::

    @article{PionTonachini2019,
      title = {ICLabel: An automated electroencephalographic independent component classifier,  dataset,  and website},
      volume = {198},
      ISSN = {1053-8119},
      url = {http://dx.doi.org/10.1016/j.neuroimage.2019.05.026},
      DOI = {10.1016/j.neuroimage.2019.05.026},
      journal = {NeuroImage},
      publisher = {Elsevier BV},
      author = {Pion-Tonachini,  Luca and Kreutz-Delgado,  Ken and Makeig,  Scott},
      year = {2019},
      month = sep,
      pages = {181–197}
    }
