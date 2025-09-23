Cite
====

If you use ``mne-icalabel``, please consider citing our paper\ :footcite:p:`Li2022`.

.. footbibliography::

.. code-block::

    @article{Li2022,
      title = {MNE-ICALabel: Automatically annotating ICA components with ICLabel in Python},
      volume = {7},
      ISSN = {2475-9066},
      url = {http://dx.doi.org/10.21105/joss.04484},
      DOI = {10.21105/joss.04484},
      number = {76},
      journal = {Journal of Open Source Software},
      publisher = {The Open Journal},
      author = {Li,  Adam and Feitelberg,  Jacob and Saini,  Anand Prakash and H\"{o}chenberger, Richard and Scheltienne,  Mathieu},
      year = {2022},
      month = aug,
      pages = {4484}
    }

And the paper of the associated model.

ICLabel\ :footcite:p:`PionTonachini2019`
----------------------------------------

.. footbibliography::

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

MEGNet\ :footcite:p:`Treacher2021`
----------------------------------

.. footbibliography::

.. code-block::

    @article{TREACHER2021118402,
      title = {MEGnet: Automatic ICA-based artifact removal for MEG using spatiotemporal convolutional neural networks},
      journal = {NeuroImage},
      volume = {241},
      pages = {118402},
      year = {2021},
      issn = {1053-8119},
      doi = {https://doi.org/10.1016/j.neuroimage.2021.118402},
      url = {https://www.sciencedirect.com/science/article/pii/S1053811921006777},
      author = {Alex H. Treacher and Prabhat Garg and Elizabeth Davenport and Ryan Godwin and Amy Proskovec and Leonardo Guimaraes Bezerra and Gowtham Murugesan and Ben Wagner and Christopher T. Whitlow and Joel D. Stitzel and Joseph A. Maldjian and Albert A. Montillo},
      keywords = {MEG, Artifact, Automation, ICA, Convolutional neural network, Deep learning},
      abstract = {Magnetoencephalography (MEG) is a functional neuroimaging tool that records the magnetic fields induced by neuronal activity; however, signal from non-neuronal sources can corrupt the data. Eye-blinks, saccades, and cardiac activity are three of the most common sources of non-neuronal artifacts. They can be measured by affixing eye proximal electrodes, as in electrooculography (EOG), and chest electrodes, as in electrocardiography (ECG), however this complicates imaging setup, decreases patient comfort, and can induce further artifacts from movement. This work proposes an EOG- and ECG-free approach to identify eye-blinks, saccades, and cardiac activity signals for automated artifact suppression. The contribution of this work is three-fold. First, using a data driven, multivariate decomposition approach based on Independent Component Analysis (ICA), a highly accurate artifact classifier is constructed as an amalgam of deep 1-D and 2-D Convolutional Neural Networks (CNNs) to automate the identification and removal of ubiquitous whole brain artifacts including eye-blink, saccade, and cardiac artifacts. The specific architecture of this network is optimized through an unbiased, computer-based hyperparameter random search. Second, visualization methods are applied to the learned abstraction to reveal what features the model uses and to bolster user confidence in the model's training and potential for generalization. Finally, the model is trained and tested on both resting-state and task MEG data from 217 subjects, and achieves a new state-of-the-art in artifact detection accuracy of 98.95% including 96.74% sensitivity and 99.34% specificity on the held out test-set. This work automates MEG processing for both clinical and research use, adapts to the acquired acquisition time, and can obviate the need for EOG or ECG electrodes for artifact detection.}
    }
