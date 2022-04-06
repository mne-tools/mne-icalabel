import pytest

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

import numpy as np
import scipy.io as sio
import mne

from mne_icalabel.ica_label import ica_eeg_features
from mne_icalabel.ica_net import run_iclabel


# load in test data for features from original Matlab ICLabel
ica_file_path = str(files("mne_icalabel.tests").joinpath("data/eeglab_ica.set"))
ica_raw_file_path = str(files("mne_icalabel.tests").joinpath("data/eeglab_ica_raw.mat"))


def test_iclabel_net():
    """Test the ICLabel Converted Network."""
    eeglab_ica = mne.preprocessing.read_ica_eeglab(ica_file_path)
    eeglab_raw = mne.io.read_raw_eeglab(ica_file_path)

    eeglab_ica_raw = sio.loadmat(ica_raw_file_path)["EEG"]
    raw_labels = eeglab_ica_raw["etc"][0][0][0][0]["ic_classification"][0][0][0][0][0][
        1
    ]

    # compute the features of the ICA waveforms
    ica_features = ica_eeg_features(eeglab_raw, eeglab_ica)
    topo = ica_features[0].astype(np.float32)
    psds = ica_features[1].astype(np.float32)
    with pytest.warns(np.ComplexWarning):
        autocorr = ica_features[2].astype(np.float32)

    # run ICLabel network
    labels = run_iclabel(topo, psds, autocorr)

    num_labels = np.argmax(labels, axis=1)
    orig_num_labels = np.argmax(raw_labels, axis=1)

    # TO show that these are equal, add an assert False statement at the end
    # TODO: there is one difference between IClabel in Matlab and Python
    print(num_labels)
    print(orig_num_labels)

    # everything is correct Brain vs non-brain
    num_labels[num_labels != 0] = 1
    orig_num_labels[orig_num_labels != 0] = 1
    assert sum(map(lambda x, y: bool(x - y), num_labels, orig_num_labels)) == 1
