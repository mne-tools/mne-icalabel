:orphan:

Installation
============

Dependencies
------------

* ``mne`` (>=1.1)
* ``numpy`` (>=1.16)
* ``scipy`` (>=1.2.0)
* ``pooch`` (for example dataset access)
* ``torch`` (for running pytorch neural networks)

We require that you use Python ``3.7`` or higher.
You may choose to install ``mne-icalabel`` `via conda <#Installation via Conda>`_,
or `via Pip <#Installation via Pip>`_.

``mne-icalabel`` works best with the latest stable release of MNE-Python. To ensure
MNE-Python is up-to-date, see their `installation instructions <https://mne.tools/stable/install/index.html>`_.


Installation via Conda
----------------------

To install ``mne-icalabel`` using conda in a virtual environment,
simply run the following at the root of the repository:

.. code-block:: bash

   # with python>=3.7 at least
   conda create -n mne
   conda activate mne
   conda install -c conda-forge mne-icalabel


Installation via Pip
--------------------

To install ``mne-icalabel`` from `Pypi <https://pypi.org/project/mne-icalabel/>`_,
run the following command in the desired environment.

.. code-block:: bash

    # with python>=3.7 at least
    pip install mne-icalabel

Note that you can install extra dependencies with keywords:

.. code-block:: bash

    # If you would like GUI functionalities
    pip install mne-icalabel[gui]

    # If you would like MNE's ICA dependencies
    pip install mne-icalabel[ica]

    # If you are a developer and would like to install the developer dependencies
    pip install mne-icalabel[doc,style,test]

    # If you would like full functionality, which installs all of the above
    pip install mne-icalabel[all]

If you want to install a snapshot of the current development version, run:

.. code-block:: bash

   pip install git+https://github.com/mne-tools/mne-icalabel

To check if everything worked fine, the following command should not give any
error messages:

.. code-block:: bash

   python -c 'import mne_icalabel'

Installation via MNE-Installer
------------------------------

Since MNE v1.0, there is now a standalone MNE installer, which can also optionally install
``mne-icalabel``! See `MNE page <https://mne.tools/stable/install/installers.html>`_ for more information.
