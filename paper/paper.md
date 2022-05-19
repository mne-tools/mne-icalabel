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

Scalp electroencephalography (EEG) and magnetoencephalography (MEG) analysis is typically very noisy and contains various non-neural signals, such as heart beat artifacts. Independent component analysis (ICA) is a common procedure to remove these artifacts [@Bell1995]. However, removing artifacts requires manual annotation of ICA components, which is subject to human error and very laborious when operating on large datasets. This work adds the popular ICLabel model [@iclabel2019] to the MNE-Python [@Agramfort2013] software toolkit. This enables the automatic labeling and annotation of ICA components improving the preprocessing and analysis pipeline of electrophysiological data.

The Python ICLabel model is fully tested against and matches exactly the output produced in its MATLAB counterpart [@iclabel2019]. Moreover, this work builds the API on top of the robust MNE-Python ecosystem, enabling a seamless integration of automatic ICA analysis.

# Statement of need

Typically EEG and MEG data has many artifacts due to a variety of reasons, such as heart beats, eye blinks, line noise, channel noise, or muscle artifacts. A common signal processing technique for assisting in separating true brain signal from these types of signals is ICA [@Bell1995]. ICA performs blind-source separation decomposing the observed noisy electrophysiological signals into "sources", which are maximally independent. For example, this then allows researchers to decompose EEG signals into brain signals and a heart beat signal. Then by manually annotating a component time-series as a heart beat, one can remove the heart beat source from the signal and obtain their signals without the heart beat. This process typically is manually carried out and requires a human to label each component that comes out of ICA. This is subject to human error and is difficult to scale up when dealing with high-dimensional EEG, or MEG recordings.

ICLabel was a proposed statistical model that uses neural networks and a crowdsourced training dataset to automatically label ICA components [@iclabel2019]. However, this package and functionality was previously only available in MATLAB, limiting its usage among Python neuroscience researchers. Moreover, the package relied on an outdated version of Tensorflow, which makes the model difficult to build on. In this work, we will build on top of the ICLabel model to improve its robustness when operating on different types of recording hardware and channel count. Moreover, we plan on building new types of models that improve its overall accuracy and performance with respect to auto-labeling neural and non-neural signals. The availability of a simple API for ICLabel will facilitate a strong benchmark to build future models and bring in new developers and contributors.

MNE-Python is a general-purpose electrophysiology analysis package in Python that has a large core group of developers. Integration into MNE makes it likely that MNE-ICALabel will be maintained and continue to be improved. MNE-Python also has stability due to the funding it receives directly for development from institutions such as National Institutes of Health, the Chan Zuckerberg open-source initiative, the European Research Council and Agence Nationale de la Recherche in France.

The developer team is excited to improve the state of the art in data handling and looking forward to welcoming new contributors and users.

# Acknowledgements

``MNE-ICALabel`` development is partly supported by
the National Science Foundation under Grant # 2030859 to the Computing Research Association for the CIFellows Project.

We acknowledge the work of [@iclabel2019], which was originally produced in MATLAB in the popular EEGLab package.

# References
