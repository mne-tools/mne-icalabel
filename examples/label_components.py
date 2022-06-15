# -*- coding: utf-8 -*-
"""
.. _tut-label-ica-components:

Labeling ICA components with a GUI
==================================

This tutorial covers how to label ICA components with a GUI.
"""

# %%

import os

import mne
from mne.preprocessing import ICA

import mne_icalabel

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file)

# Here we'll crop to 60 seconds and drop gradiometer channels for speed
raw.crop(tmax=60.0).pick_types(meg="mag", eeg=True, stim=True, eog=True)
raw.load_data()

# high-pass filter the data and then perform ICA
filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
ica = ICA(n_components=15, max_iter="auto", random_state=97)
ica.fit(filt_raw)

# now label
gui = mne_icalabel.gui.label_ica_components(raw, ica)

# The `ica` object is modified to contain the component labels
# after closing the GUI and can now be saved
# gui.close()  # typically you close when done

# Now, we can take a look at the components, which can be
# saved into the BIDs directory.
