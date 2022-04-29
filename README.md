# ICLabel-Python

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Codecov](https://codecov.io/gh/adam2392/iclabel-python/branch/main/graph/badge.svg)](https://codecov.io/gh/adam2392/iclabel-python)
[![unit_tests](https://github.com/adam2392/iclabel-python/actions/workflows/unit_tests.yml/badge.svg?branch=main)](https://github.com/adam2392/iclabel-python/actions/workflows/unit_tests.yml)
[![CircleCI](https://circleci.com/gh/adam2392/iclabel-python.svg?style=shield)](https://circleci.com/gh/adam2392/iclabel-python)
[![PyPI Download count](https://pepy.tech/badge/mne-icalabel)](https://pepy.tech/project/mne-icalabel)
[![Latest PyPI release](https://img.shields.io/pypi/v/mne-icalabel.svg)](https://pypi.org/project/mne-icalabel/)

This repository is a conversion of the popular ICLabel classifier for Python. In addition, we provide improvements in the form of other models.

# Why?

Scalp EEG is inherently noisy comprised commonly with heartbeat, eyeblink, muscle and movement artifacts. Independent component analysis (ICA) is a common method to remove artifacts, but rely on a human manually annotating with independent components (IC) are noisy and which are brain signal.

This package aims at automating that process conforming to the popular MNE-Python API for EEG, MEG and iEEG data.

# Basic Usage

MNE-ICALabel will estimate the labels of the ICA components given
a MNE-Python object and an ICA instance using the [ICA decomposition](https://mne.tools/stable/generated/mne.preprocessing.ICA.html) available
in MNE-Python.

```Python
    from mne_icalabel import label_components

    # assuming you have a Raw loaded and ICA instance previously ran
    label_components(raw, ica)
```

# Documentation
[Stable version](https://mne.tools/mne-icalabel/stable/index.html) documentation.
[Dev version](https://mne.tools/mne-icalabel/dev/index.html) documentation.

# Installation

To get the latest code using [git](https://git-scm.com/), open a terminal and type:

    git clone git://github.com/mne-tools/mne-icalabel.git
    cd iclabel-python
    pip install -e .    

or one can install directly using pip

    pip install --user -U https://api.github.com/repos/mne-tools/mne-icalabel/zipball/main

Alternatively, you can also download a
`zip file of the latest development version <https://github.com/mne-tools/mne-icalabel/archive/main.zip>`_.

# Contributing

If you are interested in contributing, please read the [contributing guidelines](https://github.com/mne-tools/mne-icalabel/main/CONTRIBUTING.md).


# Forum

Please visit the MNE forum to ask relevant questions.

https://mne.discourse.group