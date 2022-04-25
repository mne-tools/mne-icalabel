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
    _next_power_of_2,
    _eeg_rpsd_constants,
    _eeg_rpsd_compute_psdmed,
    _eeg_rpsd_format,
    eeg_autocorr_welch,
    eeg_autocorr,
    eeg_autocorr_fftw,
)
from mne_icalabel.utils import mne_to_eeglab_locs


# Raw/Epochs files with ICA decomposition
raw_eeglab_path = str(
    files("mne_icalabel.tests").joinpath("data/datasets/sample-raw.set")
)
raw_short_eeglab_path = str(
    files("mne_icalabel.tests").joinpath("data/datasets/sample-short-raw.set")
)
raw_very_short_eeglab_path = str(
    files("mne_icalabel.tests").joinpath("data/datasets/sample-very-short-raw.set")
)
epo_eeglab_path = str(
    files("mne_icalabel.tests").joinpath("data/datasets/sample-epo.set")
)

# ICA activation matrix for raw/epochs
raw_icaact_eeglab_path = str(
    files("mne_icalabel.tests").joinpath("data/icaact/icaact-raw.mat")
)
epo_icaact_eeglab_path = str(
    files("mne_icalabel.tests").joinpath("data/icaact/icaact-epo.mat")
)

# Topography
loc_raw_path = str(files("mne_icalabel.tests").joinpath("data/topo/loc-raw.mat"))
loc_epo_path = str(files("mne_icalabel.tests").joinpath("data/topo/loc-raw.mat"))

# PSD
psd_constants_raw_path = str(
    files("mne_icalabel.tests").joinpath("data/psd/constants-raw.mat")
)
psd_psdmed_raw_path = str(
    files("mne_icalabel.tests").joinpath("data/psd/psdmed-raw.mat")
)
psd_steps_raw_path = str(
    files("mne_icalabel.tests").joinpath("data/psd/psd-step-by-step-raw.mat")
)
psd_raw_path = str(files("mne_icalabel.tests").joinpath("data/psd/psd-raw.mat"))
psd_constants_epo_path = str(
    files("mne_icalabel.tests").joinpath("data/psd/constants-epo.mat")
)
psd_psdmed_epo_path = str(
    files("mne_icalabel.tests").joinpath("data/psd/psdmed-epo.mat")
)
psd_steps_epo_path = str(
    files("mne_icalabel.tests").joinpath("data/psd/psd-step-by-step-epo.mat")
)
psd_epo_path = str(files("mne_icalabel.tests").joinpath("data/psd/psd-epo.mat"))

# Autocorrelations
autocorr_raw_path = autocorr_short_raw_path = str(
    files("mne_icalabel.tests").joinpath("data/autocorr/autocorr-raw.mat")
)
autocorr_short_raw_path = str(
    files("mne_icalabel.tests").joinpath("data/autocorr/autocorr-short-raw.mat")
)
autocorr_very_short_raw_path = str(
    files("mne_icalabel.tests").joinpath("data/autocorr/autocorr-very-short-raw.mat")
)
autocorr_epo_path = str(
    files("mne_icalabel.tests").joinpath("data/autocorr/autocorr-epo.mat")
)


# ----------------------------------------------------------------------------
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
    raw = read_raw(raw_eeglab_path, preload=True)
    ica = read_ica_eeglab(raw_eeglab_path)
    icaact = compute_ica_activations(raw, ica)

    icaact_eeglab = loadmat(raw_icaact_eeglab_path)["icaact"]
    assert np.allclose(icaact, icaact_eeglab, atol=1e-4)

    # Epoch instance
    epochs = read_epochs_eeglab(epo_eeglab_path)
    ica = read_ica_eeglab(epo_eeglab_path)
    icaact = compute_ica_activations(epochs, ica)

    icaact_eeglab = loadmat(epo_icaact_eeglab_path)["icaact"]
    assert np.allclose(icaact, icaact_eeglab, atol=1e-4)


# ----------------------------------------------------------------------------
def test_loc():
    """Test conversion of MNE montage to EEGLAB loc. 
    
    This test works because
    MNE does the conversion from EEGLAB to MNE montage when loading the
    datasets."""
    # from raw
    raw = read_raw(raw_eeglab_path, preload=True)
    rd, th = mne_to_eeglab_locs(raw)
    eeglab_loc = loadmat(loc_raw_path)["loc"][0, 0]
    eeglab_rd = eeglab_loc["rd"]
    eeglab_th = eeglab_loc["th"]
    assert np.allclose(rd, eeglab_rd, atol=1e-8)
    assert np.allclose(th, eeglab_th, atol=1e-8)

    # from epochs
    epochs = read_epochs_eeglab(epo_eeglab_path)
    rd, th = mne_to_eeglab_locs(epochs)
    eeglab_loc = loadmat(loc_epo_path)["loc"][0, 0]
    eeglab_rd = eeglab_loc["rd"]
    eeglab_th = eeglab_loc["th"]
    assert np.allclose(rd, eeglab_rd, atol=1e-8)
    assert np.allclose(th, eeglab_th, atol=1e-8)


# ----------------------------------------------------------------------------
def test_eeg_rpsd_constants():
    """Test _eeg_rpsd_constants function."""
    # Raw --------------------------------------------------------------------
    raw = read_raw(raw_eeglab_path, preload=True)
    ica = read_ica_eeglab(raw_eeglab_path)
    ncomp, nfreqs, n_points, nyquist, index, window, subset = _eeg_rpsd_constants(
        raw, ica
    )

    constants_eeglab = loadmat(psd_constants_raw_path)["constants"][0, 0]
    ncomp_eeglab = constants_eeglab["ncomp"][0, 0]
    nfreqs_eeglab = constants_eeglab["nfreqs"][0, 0]
    n_points_eeglab = constants_eeglab["n_points"][0, 0]
    nyquist_eeglab = constants_eeglab["nyquist"][0, 0]
    index_eeglab = constants_eeglab["index"]
    window_eeglab = constants_eeglab["window"]
    subset_eeglab = constants_eeglab["subset"]

    assert ncomp == ncomp_eeglab
    assert nfreqs == nfreqs_eeglab
    assert n_points == n_points_eeglab
    assert nyquist == nyquist_eeglab
    assert np.allclose(index, index_eeglab - 1)

    # window and subset are not squeezed in matlab
    assert window_eeglab.shape[0] == 1
    assert np.allclose(window, window_eeglab[0, :])

    # for subsets, compare if the same elements are in both
    assert subset_eeglab.shape[0] == 1
    assert len(set(list(subset)).difference(set(list(subset_eeglab[0, :] - 1)))) == 0
    assert len(set(list(subset_eeglab[0, :] - 1)).difference(set(list(subset)))) == 0

    # Epochs -----------------------------------------------------------------
    epochs = read_epochs_eeglab(epo_eeglab_path)
    ica = read_ica_eeglab(epo_eeglab_path)
    ncomp, nfreqs, n_points, nyquist, index, window, subset = _eeg_rpsd_constants(
        epochs, ica
    )

    constants_eeglab = loadmat(psd_constants_epo_path)["constants"][0, 0]
    ncomp_eeglab = constants_eeglab["ncomp"][0, 0]
    nfreqs_eeglab = constants_eeglab["nfreqs"][0, 0]
    n_points_eeglab = constants_eeglab["n_points"][0, 0]
    nyquist_eeglab = constants_eeglab["nyquist"][0, 0]
    index_eeglab = constants_eeglab["index"]
    window_eeglab = constants_eeglab["window"]
    subset_eeglab = constants_eeglab["subset"]

    assert ncomp == ncomp_eeglab
    assert nfreqs == nfreqs_eeglab
    assert n_points == n_points_eeglab
    assert nyquist == nyquist_eeglab
    assert np.allclose(index, index_eeglab - 1)

    # window and subset are not squeezed in matlab
    assert window_eeglab.shape[0] == 1
    assert np.allclose(window, window_eeglab[0, :])

    # for subsets, compare if the same elements are in both
    assert subset_eeglab.shape[0] == 1
    assert len(set(list(subset)).difference(set(list(subset_eeglab[0, :] - 1)))) == 0
    assert len(set(list(subset_eeglab[0, :] - 1)).difference(set(list(subset)))) == 0


def test_eeg_rpsd_compute_psdmed():
    """Test _eeg_rpsd_compute_psdmed function."""
    # Raw --------------------------------------------------------------------
    raw = read_raw(raw_eeglab_path, preload=True)
    ica = read_ica_eeglab(raw_eeglab_path)
    icaact = compute_ica_activations(raw, ica)

    # retrieve subset from eeglab
    constants_eeglab = loadmat(psd_constants_raw_path)["constants"][0, 0]
    assert constants_eeglab["subset"].shape[0] == 1
    subset_eeglab = constants_eeglab["subset"][0, :] - 1

    # retrieve the rest from python
    ncomp, nfreqs, n_points, nyquist, index, window, _ = _eeg_rpsd_constants(raw, ica)

    # compute psdmed
    psdmed = _eeg_rpsd_compute_psdmed(
        raw, icaact, ncomp, nfreqs, n_points, nyquist, index, window, subset_eeglab
    )

    psdmed_eeglab = loadmat(psd_psdmed_raw_path)["psdmed"]
    assert np.allclose(psdmed, psdmed_eeglab, atol=1e-4)

    # Epochs -----------------------------------------------------------------
    epochs = read_epochs_eeglab(epo_eeglab_path)
    ica = read_ica_eeglab(epo_eeglab_path)
    icaact = compute_ica_activations(epochs, ica)

    # retrieve subset from eeglab
    constants_eeglab = loadmat(psd_constants_epo_path)["constants"][0, 0]
    assert constants_eeglab["subset"].shape[0] == 1
    subset_eeglab = constants_eeglab["subset"][0, :] - 1

    # retrieve the rest from python
    ncomp, nfreqs, n_points, nyquist, index, window, _ = _eeg_rpsd_constants(
        epochs, ica
    )

    # compute psdmed
    psdmed = _eeg_rpsd_compute_psdmed(
        epochs, icaact, ncomp, nfreqs, n_points, nyquist, index, window, subset_eeglab
    )

    psdmed_eeglab = loadmat(psd_psdmed_epo_path)["psdmed"]
    assert np.allclose(psdmed, psdmed_eeglab, atol=5e-2)
    # TODO: investigate why the tolerance had to be brought this high for this
    # particular case.


def test_eeg_rpsd():
    """Test eeg_rpsd function that extract the PSD feature from the IC."""
    # Raw --------------------------------------------------------------------
    # Compare that both MATLAB files are identical (since rng('default') was
    # called both time, resetting the seed).
    psd1 = loadmat(psd_steps_raw_path)["psd"]
    psd2 = loadmat(psd_raw_path)["psd"]
    assert np.allclose(psd1, psd2, atol=1e-4)

    # clean-up
    psd_eeglab = psd2.copy()
    del psd1
    del psd2

    # compute psd in Python
    raw = read_raw(raw_eeglab_path, preload=True)
    ica = read_ica_eeglab(raw_eeglab_path)
    icaact = compute_ica_activations(raw, ica)

    # retrieve subset from eeglab
    constants_eeglab = loadmat(psd_constants_raw_path)["constants"][0, 0]
    assert constants_eeglab["subset"].shape[0] == 1
    subset_eeglab = constants_eeglab["subset"][0, :] - 1

    # retrieve the rest from python
    ncomp, nfreqs, n_points, nyquist, index, window, _ = _eeg_rpsd_constants(raw, ica)

    # compute psdmed
    psdmed = _eeg_rpsd_compute_psdmed(
        raw, icaact, ncomp, nfreqs, n_points, nyquist, index, window, subset_eeglab
    )

    # format and compare
    psd = _eeg_rpsd_format(psdmed)
    assert np.allclose(psd, psd_eeglab, atol=1e-4)

    # Epochs -----------------------------------------------------------------
    # Compare that both MATLAB files are identical (since rng('default') was
    # called both time, resetting the seed).
    psd1 = loadmat(psd_steps_epo_path)["psd"]
    psd2 = loadmat(psd_epo_path)["psd"]
    assert np.allclose(psd1, psd2, atol=1e-4)

    # clean-up
    psd_eeglab = psd2.copy()
    del psd1
    del psd2

    # compute psd in Python
    epochs = read_epochs_eeglab(epo_eeglab_path)
    ica = read_ica_eeglab(epo_eeglab_path)
    icaact = compute_ica_activations(epochs, ica)

    # retrieve subset from eeglab
    constants_eeglab = loadmat(psd_constants_epo_path)["constants"][0, 0]
    assert constants_eeglab["subset"].shape[0] == 1
    subset_eeglab = constants_eeglab["subset"][0, :] - 1

    # retrieve the rest from python
    ncomp, nfreqs, n_points, nyquist, index, window, _ = _eeg_rpsd_constants(
        epochs, ica
    )

    # compute psdmed
    psdmed = _eeg_rpsd_compute_psdmed(
        epochs, icaact, ncomp, nfreqs, n_points, nyquist, index, window, subset_eeglab
    )

    # format and compare
    psd = _eeg_rpsd_format(psdmed)
    assert np.allclose(psd, psd_eeglab, atol=1e-5)


# ----------------------------------------------------------------------------
def test_next_power_of_2():
    """Test that next_power_of_2 works as intended."""
    x = [0, 10, 200, 400]
    expected = [1, 16, 256, 512]
    for k, exp in zip(x, expected):
        val = _next_power_of_2(k)
        assert exp == val


def test_eeg_autocorr_welch():
    """Test eeg_autocorr_welch feature used on long raw datasets."""
    raw = read_raw(raw_eeglab_path, preload=True)
    ica = read_ica_eeglab(raw_eeglab_path)
    icaact = compute_ica_activations(raw, ica)
    autocorr = eeg_autocorr_welch(raw, ica, icaact)

    autocorr_eeglab = loadmat(autocorr_raw_path)["autocorr"]
    assert np.allclose(autocorr, autocorr_eeglab, atol=1e-4)


def test_eeg_autocorr():
    """Test eeg_autocorr feature used on short raw datasets."""
    # Raw between 1 and 5 seconds
    raw = read_raw(raw_short_eeglab_path, preload=True)
    ica = read_ica_eeglab(raw_short_eeglab_path)
    icaact = compute_ica_activations(raw, ica)
    autocorr = eeg_autocorr(raw, ica, icaact)

    autocorr_eeglab = loadmat(autocorr_short_raw_path)["autocorr"]
    assert np.allclose(autocorr, autocorr_eeglab, atol=1e-4)

    # Raw shorter than 1 second
    raw = read_raw(raw_very_short_eeglab_path, preload=True)
    ica = read_ica_eeglab(raw_very_short_eeglab_path)
    icaact = compute_ica_activations(raw, ica)
    autocorr = eeg_autocorr(raw, ica, icaact)

    autocorr_eeglab = loadmat(autocorr_very_short_raw_path)["autocorr"]
    assert np.allclose(autocorr, autocorr_eeglab, atol=1e-4)


def test_eeg_autocorr_fftw():
    """Test eeg_autocorr_fftw feature used on epoch datasets."""
    epochs = read_epochs_eeglab(epo_eeglab_path)
    ica = read_ica_eeglab(epo_eeglab_path)
    icaact = compute_ica_activations(epochs, ica)
    autocorr = eeg_autocorr_fftw(epochs, ica, icaact)

    autocorr_eeglab = loadmat(autocorr_epo_path)["autocorr"]
    assert np.allclose(autocorr, autocorr_eeglab, atol=1e-4)
