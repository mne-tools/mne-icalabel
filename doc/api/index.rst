API
===

This is the application programming interface (API) reference
for classes (``CamelCase`` names) and functions
(``underscore_case`` names) of MNE-ICALabel.

Most-used function
------------------

The most commonly used function is :func:`mne_icalabel.label_components` which takes an
mne instance (:class:`mne.io.Raw`, :class:`mne.Epochs`) and its ICA decomposition to
label the components using the specified method/model.

.. currentmodule:: mne_icalabel

.. autosummary::
    :toctree: ../generated/api

    label_components

Models
------

.. card-carousel:: 2

    .. card:: ICLabel
        :link: iclabel.html
        :link-type: url

Features
--------

On top of the available models, ``mne-icalabel`` provides a set of functions to extract
features from `~mne.preprocessing.ICA` instance and `~mne.io.Raw` / `~mne.Epochs`
instances using MNE-Python. Those features can then be used to train new models.

.. currentmodule:: mne_icalabel.features

.. autosummary::
    :toctree: ../generated/api

    get_topomaps

Annotating Components
---------------------

Finally, to facilitate annotation of the ICA components, we provide an API that conforms
to the derivative standard of BIDS for EEG data to write the annotations to a TSV file.

.. currentmodule:: mne_icalabel.annotation

.. autosummary::
    :toctree: ../generated/api

    mark_component
    write_components_tsv

In addition, as of version 0.3, we have introduced a beta-version of a GUI that
assists in annotated ICA components. This was heavily inspired by the annotation
process in ``ICLabel``.

.. currentmodule:: mne_icalabel.gui

.. autosummary::
    :toctree: ../generated/api

    label_ica_components
