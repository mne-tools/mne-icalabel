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

from mne_icalabel.gui import label_ica_components

# %%
# Load in some sample data

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file)

# Here we'll crop to 60 seconds and drop gradiometer channels for speed
raw.crop(tmax=60.0).pick_types(meg="mag", eeg=True, stim=True, eog=True)
raw.load_data()

# %%
# Preprocess and run ICA on the data
# ----------------------------------
# Before labeling components with the GUI, one needs to filter the data
# and then fit the ICA instance. Afterwards, one can run the GUI using the
# ``Raw`` data object and the fitted ``ICA`` instance. The GUI will modify
# the ICA instance in place, and add the labels of each component to
# the ``labels_`` attribute.

# high-pass filter the data and then perform ICA
filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
ica = ICA(n_components=15, max_iter="auto", random_state=97)
ica.fit(filt_raw)

# now label the components using a GUI
mne.set_log_level("DEBUG")
gui = label_ica_components(raw, ica)

# The `ica` object is modified to contain the component labels
# after closing the GUI and can now be saved
# gui.close()  # typically you close when done

# Now, we can take a look at the components, which can be
# saved into the BIDs directory.
print(ica.labels_)

# %%
# Save the labeled components
# ---------------------------
# After the GUI labels, save the components using the ``write_components_tsv``
# function. This will save the ICA annotations to disc in BIDS-Derivative for
# EEG data format.
#
# Note: BIDS-EEG-Derivatives is not fully specified, so this functionality
# may change in the future without notice.

# fname = '<some path to save the components>'
# write_components_tsv(ica, fname)
