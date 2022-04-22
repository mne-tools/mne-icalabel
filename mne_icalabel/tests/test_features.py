try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from mne import read_epochs_eeglab
from mne.io import read_raw
from mne.io.eeglab.eeglab import _check_load_mat
from mne.preprocessing import read_ica_eeglab
import numpy as np
from scipy.io import loadmat

from mne_icalabel.features import (
    retrieve_eeglab_icawinv,
    compute_ica_activations,
    next_power_of_2,
    eeg_autocorr,
    )


# Raw/Epochs files with ICA decomposition
raw_eeglab_path = str(files("mne_icalabel.tests").joinpath("data/datasets/sample-raw.set"))
raw_short_eeglab_path = str(files("mne_icalabel.tests").joinpath("data/datasets/sample-short-raw.set"))
epo_eeglab_path = str(files("mne_icalabel.tests").joinpath("data/datasets/sample-epo.set"))

# ICA activation matrix for raw/epochs
raw_icaact_eeglab_path = str(
    files("mne_icalabel.tests").joinpath("data/icaact/icaact-raw.mat")
)
epo_icaact_eeglab_path = str(
    files("mne_icalabel.tests").joinpath("data/icaact/icaact-epo.mat")
)

# Autocorrelations
autocorr_short_raw = str(
    files("mne_icalabel.tests").joinpath("data/autocorr/autocorr-short-raw.mat")
)


def test_retrieve_eeglab_icawinv():
    """Test that the icawinv is correctly retrieved from an MNE ICA object."""
    # Raw instance
    ica = read_ica_eeglab(raw_eeglab_path)
    icawinv, _ = retrieve_eeglab_icawinv(ica)

    eeg = _check_load_mat(raw_eeglab_path, None)
    assert np.allclose(icawinv, eeg.icawinv)

    # Epoch instance
    ica = read_ica_eeglab(epo_eeglab_path)
    icawinv, _ = retrieve_eeglab_icawinv(ica)
    eeg = _check_load_mat(epo_eeglab_path, None)
    assert np.allclose(icawinv, eeg.icawinv)


def test_compute_ica_activations():
    """Test that the icaact is correctly retrieved from an MNE ICA object."""
    # Raw instance
    raw = read_raw(raw_eeglab_path)
    ica = read_ica_eeglab(raw_eeglab_path)
    icaact = compute_ica_activations(raw, ica)

    icaact_eeglab = loadmat(raw_icaact_eeglab_path)["icaact"]
    assert np.allclose(icaact, icaact_eeglab, rtol=1e-8, atol=1e-4)

    # Epoch instance
    epochs = read_epochs_eeglab(epo_eeglab_path)
    ica = read_ica_eeglab(epo_eeglab_path)
    icaact = compute_ica_activations(epochs, ica)

    icaact_eeglab = loadmat(epo_icaact_eeglab_path)["icaact"]
    assert np.allclose(icaact, icaact_eeglab, rtol=1e-8, atol=1e-4)


def test_next_power_of_2():
    """Test that next_power_of_2 works as intended."""
    x = [0, 10, 200, 400]
    expected = [1, 16, 256, 512]
    for k, exp in zip(x, expected):
        val = next_power_of_2(k)
        assert exp == val


def test_eeg_autocorr():
    """Test eeg_autocorr feature used on short raw datasets."""
    raw = read_raw(raw_short_eeglab_path)
    ica = read_ica_eeglab(raw_short_eeglab_path)
    icaact = compute_ica_activations(raw, ica)
    autocorr = eeg_autocorr(raw, ica, icaact)

    autocorr_eeglab = loadmat(autocorr_short_raw)["autocorr"]
    assert np.allclose(autocorr, autocorr_eeglab, atol=1e-4)

    # TODO: Add test with a raw that is shorter than 1 second.
