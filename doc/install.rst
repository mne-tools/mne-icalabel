:orphan:

Installation
============

Dependencies
------------

* ``mne`` (>=1.0)
* ``numpy`` (>=1.16)
* ``scipy`` (>=1.2.0)
* ``torch`` (for running pytorch neural networks)

We require that you use Python 3.7 or higher.
You may choose to install ``mne-icalabel`` `via pip <#Installation via pip>`_,
or conda.

Installation via Conda
----------------------

To install mne-icalabel using conda in a virtual environment,
simply run the following at the root of the repository:

.. code-block:: bash

   # with python>=3.8 at least
   conda create -n mne
   conda activate mne
   conda install -c conda-forge mne-icalabel


Installation via Pip
--------------------

To install mne-icalabel including all dependencies required to use all features,
simply run the following at the root of the repository:

.. code-block:: bash

    python -m venv .venv
    pip install -U mne-icalabel

If you want to install a snapshot of the current development version, run:

.. code-block:: bash

   pip install --user -U https://api.github.com/repos/mne-tools/mne-icalabel/zipball/main

To check if everything worked fine, the following command should not give any
error messages:

.. code-block:: bash

   python -c 'import mne_icalabel'

mne-icalabel works best with the latest stable release of MNE-Python. To ensure
MNE-Python is up-to-date, run:

.. code-block:: bash

   pip install --user -U mne
