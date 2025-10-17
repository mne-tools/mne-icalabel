.. include:: ./links.inc

Installation
============

``mne-icalabel`` requires Python ``3.9`` or higher.

Dependencies
------------

``mne-icalabel`` works best with the latest stable release of MNE-Python. To
ensure MNE-Python is up-to-date, see
`MNE installation instructions <mne install_>`_.
``mne-icalabel`` is available on `Pypi <project pypi_>`_ and
on `conda-forge <project conda_>`_.

Methods
-------

.. tab-set::

    .. tab-item:: MNE installers

        As of MNE-Python 1.0, ``mne-icalabel`` is distributed in the
        `MNE standalone installers <mne installers_>`_.

        The installers create a conda environment with the entire MNE-ecosystem
        setup, and more!

    .. tab-item:: PyPI

        ``mne-icalabel`` is available on `PyPI <project pypi_>`_ and can be
        installed in a given environment via ``pip``.

        .. code-block:: bash

            pip install mne-icalabel

        The ICLabel model requires either `pytorch <https://pytorch.org/>`_ or
        `Microsoft onnxruntime <https://onnxruntime.ai/>`_.

        .. code-block:: bash

            pip install torch
            pip install onnxruntime
        
        .. note::

            If you are working with MEG data and plan to use the MEGnet model, e.g.
            :func:`mne_icalabel.megnet.megnet_label_components`, you *must* install
            ``onnxruntime``, and do not need to install ``torch``.

        Additional dependencies can be installed with different keywords:

        .. code-block:: bash

            # GUI functionalities
            pip install mne-icalabel[gui]

            # MNE's ICA dependencies
            pip install mne-icalabel[ica]

            # developer dependencies
            pip install mne-icalabel[doc,stubs,style,test]

            # all of the above
            pip install mne-icalabel[all]

    .. tab-item:: Conda

        Depending on your system, you may want to create a separate environment
        to install ``mne-icalabel``. You can create a virtual environment with
        conda:

        .. code-block:: bash

            conda create -n myenv
            conda activate myenv

        Replace ``myenv`` with the environment name you prefer.
        ``mne-icalabel`` can then be installed from the
        `conda-forge <project conda_>`_ channel:

        .. code-block:: bash

            conda install -c conda-forge mne-icalabel

    .. tab-item:: Source

        If you want to install a snapshot of the current development version,
        run:

        .. code-block:: bash

            pip install git+https://github.com/mne-tools/mne-icalabel

To check if everything worked fine, the following command should not raise any
error messages:

.. code-block:: bash

    mne_icalabel-sys_info
