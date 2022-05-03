# -*- coding: utf-8 -*-
"""
.. _tut-auto-artifact-ica:

Repairing artifacts with ICA automatically using ICLabel Model
==============================================================

This tutorial covers automatically repairing signals using ICA with
the ICLabel model :footcite:`iclabel2019`, which originates in EEGLab.
For conceptual background on ICA, see :ref:`this scikit-learn tutorial
<sphx_glr_auto_examples_decomposition_plot_ica_blind_source_separation.py>`.
For a basic understanding of how to use ICA to remove artifacts, see `the
tutorial <https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html>`_ in MNE-Python.

We begin as always by importing the necessary Python modules and loading some
:ref:`example data <sample-dataset>`. Because ICA can be computationally
intense, we'll also crop the data to 60 seconds; and to save ourselves from
repeatedly typing ``mne.preprocessing`` we'll directly import a few functions
and classes from that submodule.
"""

# %%

import os

import mne
from mne.preprocessing import ICA

from mne_icalabel import label_components

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file)

# Here we'll crop to 60 seconds and drop gradiometer channels for speed
raw.crop(tmax=60.0).pick_types(meg="mag", eeg=True, stim=True, eog=True)
raw.load_data()

# %%
# .. note::
#     Before applying ICA (or any artifact repair strategy), be sure to observe
#     the artifacts in your data to make sure you choose the right repair tool.
#     Sometimes the right tool is no tool at all — if the artifacts are small
#     enough you may not even need to repair them to get good analysis results.
#     See :ref:`tut-artifact-overview` for guidance on detecting and
#     visualizing various types of artifact.
#

# Example: EOG and ECG artifact repair
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Visualizing the artifacts
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's begin by visualizing the artifacts that we want to repair. In this
# dataset they are big enough to see easily in the raw data:

# Note: for this example, we are using ICLabel which has only
# been validated and works for EEG systems with less than 32 electrodes.
raw = raw.pick_types(eeg=True, eog=True, ecg=True, emg=True)

# pick some channels that clearly show heartbeats and blinks
regexp = r"(EEG 00.)"
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)

# %%
# Filtering to remove slow drifts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Before we run the ICA, an important step is filtering the data to remove
# low-frequency drifts, which can negatively affect the quality of the ICA fit.
# The slow drifts are problematic because they reduce the independence of the
# assumed-to-be-independent sources (e.g., during a slow upward drift, the
# neural, heartbeat, blink, and other muscular sources will all tend to have
# higher values), making it harder for the algorithm to find an accurate
# solution. A high-pass filter with 1 Hz cutoff frequency is recommended.
# However, because filtering is a linear operation, the ICA solution found from
# the filtered signal can be applied to the unfiltered signal (see
# :footcite:`WinklerEtAl2015` for
# more information), so we'll keep a copy of the unfiltered
# `~mne.io.Raw` object around so we can apply the ICA solution to it
# later.

filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)

# %%
# Fitting and plotting the ICA solution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. sidebar:: Ignoring the time domain
#
#     The ICA algorithms implemented in MNE-Python find patterns across
#     channels, but ignore the time domain. This means you can compute ICA on
#     discontinuous `~mne.Epochs` or `~mne.Evoked` objects (not
#     just continuous `~mne.io.Raw` objects), or only use every Nth
#     sample by passing the ``decim`` parameter to ``ICA.fit()``.
#
#     .. note:: `~mne.Epochs` used for fitting ICA should not be
#               baseline-corrected. Because cleaning the data via ICA may
#               introduce DC offsets, we suggest to baseline correct your data
#               **after** cleaning (and not before), should you require
#               baseline correction.
#
# Now we're ready to set up and fit the ICA. Since we know (from observing our
# raw data) that the EOG and ECG artifacts are fairly strong, we would expect
# those artifacts to be captured in the first few dimensions of the PCA
# decomposition that happens before the ICA. Therefore, we probably don't need
# a huge number of components to do a good job of isolating our artifacts
# (though it is usually preferable to include more components for a more
# accurate solution). As a first guess, we'll run ICA with ``n_components=15``
# (use only the first 15 PCA components to compute the ICA decomposition) — a
# very small number given that our data has over 300 channels, but with the
# advantage that it will run quickly and we will able to tell easily whether it
# worked or not (because we already know what the EOG / ECG artifacts should
# look like).
#
# ICA fitting is not deterministic (e.g., the components may get a sign
# flip on different runs, or may not always be returned in the same order), so
# we'll also specify a `random seed`_ so that we get identical results each
# time this tutorial is built by our web servers.

ica = ICA(n_components=15, max_iter="auto", random_state=97)
ica.fit(filt_raw)
ica

# %%
# Some optional parameters that we could have passed to the
# `~mne.preprocessing.ICA.fit` method include ``decim`` (to use only
# every Nth sample in computing the ICs, which can yield a considerable
# speed-up) and ``reject`` (for providing a rejection dictionary for maximum
# acceptable peak-to-peak amplitudes for each channel type, just like we used
# when creating epoched data in the :ref:`tut-overview` tutorial).
#
# Now we can examine the ICs to see what they captured.
# `~mne.preprocessing.ICA.plot_sources` will show the time series of the
# ICs. Note that in our call to `~mne.preprocessing.ICA.plot_sources` we
# can use the original, unfiltered `~mne.io.Raw` object:

raw.load_data()
ica.plot_sources(raw, show_scrollbars=False)

# %%
# Here we can pretty clearly see that the first component (``ICA000``) captures
# the EOG signal quite well, and the second component (``ICA001``) looks a lot
# like `a heartbeat <qrs_>`_ (for more info on visually identifying Independent
# Components, `this EEGLAB tutorial`_ is a good resource). We can also
# visualize the scalp field distribution of each component using
# `~mne.preprocessing.ICA.plot_components`. These are interpolated based
# on the values in the ICA mixing matrix:
#
# .. LINKS
#
# .. _`blind source separation`:
#    https://en.wikipedia.org/wiki/Signal_separation
# .. _`statistically independent`:
#    https://en.wikipedia.org/wiki/Independence_(probability_theory)
# .. _`scikit-learn`: https://scikit-learn.org
# .. _`random seed`: https://en.wikipedia.org/wiki/Random_seed
# .. _`regular expression`: https://www.regular-expressions.info/
# .. _`qrs`: https://en.wikipedia.org/wiki/QRS_complex
# .. _`this EEGLAB tutorial`: https://labeling.ucsd.edu/tutorial/labels


# sphinx_gallery_thumbnail_number = 9
ica.plot_components()

# blinks
ica.plot_overlay(raw, exclude=[0], picks="eeg")

# %%
# We can also plot some diagnostics of each IC using
# `~mne.preprocessing.ICA.plot_properties`:

ica.plot_properties(raw, picks=[0, 1])

# Selecting ICA components automatically
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now that we've explored what components need to be removed, we can
# apply the automatic ICA component labeling algorithm, which will
# assign a probability value for each component being one of:
#
# - brain
# - muscle artifact
# - eye blink
# - heart beat
# - line noise
# - channel noise
# - other
#
# The output of the ICLabel ``label_components`` function produces
# predicted probability values for each of these classes in that order.
#
# To start this process, we will compute features of each ICA
# component to be fed into our classification model. This is
# done automatically underneath the hood. An autocorrelation,
# power spectral density and topographic map feature is fed
# into a 3-head neural network that has been pretrained.
# See :footcite:`iclabel2019` for full details.

ic_labels = label_components(raw, ica, method="iclabel")
print(ic_labels)

# We can extract the labels of each component and exclude
# non-brain classified components, keeping 'brain' and 'other'.
# "Other" is a catch-all that for non-classifiable components.
# We will ere on the side of caution and assume we cannot blindly remove these.
labels = ic_labels["labels"]
exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
print(f"Excluding these ICA components: {exclude_idx}")

# %%
# Now that the exclusions have been set, we can reconstruct the sensor signals
# with artifacts removed using the `~mne.preprocessing.ICA.apply` method
# (remember, we're applying the ICA solution from the *filtered* data to the
# original *unfiltered* signal). Plotting the original raw data alongside the
# reconstructed data shows that the heartbeat and blink artifacts are repaired.

# ica.apply() changes the Raw object in-place, so let's make a copy first:
reconst_raw = raw.copy()
ica.apply(reconst_raw, exclude=exclude_idx)

raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
reconst_raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
del reconst_raw

# %%
# References
# ^^^^^^^^^^
# .. footbibliography::
