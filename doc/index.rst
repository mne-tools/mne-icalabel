**mne-icalabel**
===================

mne-icalabel is a Python package for labeling independent components that stem
from an `Independent Component Analysis (ICA) <https://en.wikipedia.org/wiki/Independent_component_analysis>`_.

Scalp electroencephalography (EEG) and magnetoencephalography (MEG) analysis is typically very noisy
and contains various non-neural signals, such as heart beat artifacts. Independent
component analysis (ICA) is a common procedure to remove these artifacts [@Bell1995].
However, removing artifacts requires manual annotation of ICA components, which is
subject to human error and very laborious when operating on large datasets. The first
few version of MNE-ICALabel replicated the popular ICLabel model for Python (previously only available
in MATLAB's EEGLab). In future versions, the package aims to develop more robust models
that build upon the ICLabel model.

We encourage you to use the package for your research and also build on top
with relevant Pull Requests. See our examples for walk-throughs of how to use the package.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting started:

   whats_new
   install
   api
   auto_examples/index

.. toctree::
   :hidden:
   :caption: Development

   License <https://raw.githubusercontent.com/mne-tools/mne-icalabel/blob/main/LICENSE>
   Contributing <https://github.com/mne-tools/mne-icalabel/blob/main/CONTRIBUTING.md>

License
-------

**mne-icalabel** is licensed under `BSD 3.0 <https://opensource.org/licenses/BSD-3-Clause>`_.
A full copy of the license can be found `on GitHub <https://raw.githubusercontent.com/mne-tools/mne-icalabel/main/LICENSE>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
