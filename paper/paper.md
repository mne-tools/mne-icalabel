---
title: "MNE-ICALabel: Automatically annotating ICA components with ICLabel in Python"
tags:
  - Python
  - MNE
  - MEG
  - EEG
  - iEEG
  - ICA
  - EEGLab
authors:
  - affiliation: 1
    name: Adam Li
    orcid: 0000-0001-8421-365X
  - affiliation: 2
    name: Jacob Feitelberg
    orcid: tbd
  - affiliation: 3
    name: Anand Saini
    orcid: tbd
  - affiliation: 3
    name: Mathieu Scheltienne
    orcid: tbd
affiliations:
- index: 1
  name: Department of Computer Science, Columbia University, New York, United States
- index: 2
  name: Department of Biomedical Engineering, Johns Hopkins University, Baltimore, United States
- index: 3
  name: tbd
date: 18 May 2022
bibliography: paper.bib
---

# Summary

The development of the Brain Imaging Data Structure
[BIDS; @Gorgolewski2016] gave the neuroscientific community
a standard to organize and share data.
BIDS prescribes file naming conventions and a folder structure to
store data in a set of already existing file formats.
Next to rules about organization of the data itself, BIDS provides standardized
templates to store associated metadata in the form of Javascript Object
Notation (JSON) and tab separated value (TSV) files.
It thus facilitates data sharing, eases metadata querying, and enables
automatic data analysis pipelines.
BIDS is a rich system to curate, aggregate, and annotate neuroimaging
databases.

While BIDS was originally intended for magnetic resonance imaging (MRI)
data, it has extensions for other data modalities including: magnetoencephalography
[MEG; @Niso2018], electroencephalography [EEG; @Pernet2019], and
intracranial encephalography [iEEG; @Holdgraf2019].
Software packages analyzing MEG, EEG, and iEEG are
now starting to support data organized using the BIDS standard, thereby
becoming "BIDS compatible".
Within the Python ecosystem, ``MNE-Python`` [@Agramfort2013] is a major
software package for electrophysiology data analysis, and extending its
functionality to support BIDS would be a great benefit for its growing user
base.
For this reason, we developed a dedicated Python software package *``MNE-BIDS`` with
the goal of providing a programmable interface for BIDS datasets in
electrophysiology with ``MNE-Python``*.
``MNE-BIDS`` allows users to re-organize data into BIDS formats, store
associated metadata after anonymization, extract information necessary for
preprocessing, and read the data into ``MNE-Python`` objects,
ready for source localization.

Starting with a single directory full of data files with arbitrary
names, ``MNE-BIDS`` can be used to extract existing metadata, reorganize the
files into the BIDS format, and write additional metadata.
All the conversion routines are thoroughly tested by running the output through
the [BIDS validator](https://github.com/bids-standard/bids-validator).
Moreover, ``MNE-BIDS`` supports converting data formats that are not BIDS
compatible into permissible formats.
These utilities allow users to easily convert their datasets to BIDS in a
matter of minutes, rather than hours of manual labour.

In addition to this core functionality, ``MNE-BIDS`` is continuously being
extended to facilitate the analysis of BIDS formatted data.
Some features include: reading a BIDS dataset as a set of Python objects for
analysis with ``MNE-Python``,
defacing T1-weighted anatomical MRI images to anonymize data and facilitate sharing,
and saving anatomical landmark coordinates to enable
coregistration between the MEG/EEG and MRI data, which is necessary for
computation of forward and inverse solutions.

Users can easily install ``MNE-BIDS`` on all major platforms via `pip` and
`conda`, and its functionality is continuously tested on Windows, macOS, and
Linux.
Other than the core dependencies for scientific computing (`numpy`, `scipy`)
and handling of MEG/EEG data (`mne`), ``MNE-BIDS`` has minimal dependencies,
all of which are optional.
The Application Programming Interface (API) of the package is stable and
extensively documented and explained in examples
([https://mne.tools/mne-bids/](https://mne.tools/mne-bids/)).
In addition, a command-line interface is provided that allows non-Python
users to benefit from the core functionality.

As of writing, ``MNE-BIDS`` has received code contributions from 15
contributors and its user base is steadily growing.
Code development is
[active](https://github.com/mne-tools/mne-bids/graphs/commit-activity) and the
developer team is committed to provide timely support for issues opened on the
GitHub issue tracker.

``MNE-BIDS`` is used as a dependency in several other software packages such as
the [MNE-study-template](https://github.com/mne-tools/mne-study-template), an
automated pipeline for group analysis with MNE [@Mainak2018], and
[Biscuit](https://github.com/Macquarie-MEG-Research/Biscuit), a graphical
user interface to format BIDS data.
Lastly, several large institutions have adopted ``MNE-BIDS`` for their
workflows such as the Martinos Center for Biomedical Imaging.

The developer team is excited to improve the state of the art in data handling
and looking forward to welcoming new contributors and users.

# Acknowledgements

``MNE-ICALabel`` development is partly supported by
the National Science Foundation under Grant # 2030859 to the Computing Research Association for the CIFellows Project.

# References
