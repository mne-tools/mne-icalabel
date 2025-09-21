:html_theme.sidebar_secondary.remove:

.. include:: ./links.inc

**MNE-ICALabel**
================

``mne-icalabel`` is a Python package for labeling independent components that
stem from an `Independent Component Analysis (ICA) <ica wikipedia_>`_.

Scalp electroencephalography (EEG) and magnetoencephalography (MEG) analysis is
typically very noisy and contains various non-neural signals, such as heartbeat
artifacts. `Independent Component Analysis (ICA) <ica wikipedia_>`_ is a common
procedure to remove these artifacts. However, removing artifacts requires
manual annotation of ICA components, which is subject to human error and very
laborious when operating on large datasets.

The first few versions of ``mne-icalabel`` replicated the popular ICLabel model for
Python (previously only available in MATLAB's EEGLab). In future versions, the package
aims to develop more robust models that build upon the ICLabel model.

We encourage you to use the package for your research and also build on top
with relevant Pull Requests (PR). See our examples for walk-throughs of how to
use the package and see our
`contributing guide <project contributing_>`_ for contributions.

``mne-icalabel`` is licensed under the `BSD license`_.
A full copy of the license can be found `on GitHub <project license_>`_.
See our :ref:`changes/index:Changelog` for a full list of changes.

Contents
--------

.. toctree::
   :maxdepth: 2
   :hidden:

   install
   api/index
   generated/examples/index
   cite
   changes/index
