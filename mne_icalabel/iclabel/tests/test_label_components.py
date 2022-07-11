import numpy as np
import pytest
from mne import create_info, make_fixed_length_epochs, pick_types
from mne.datasets import testing
from mne.io import RawArray, read_raw
from mne.preprocessing import ICA

from mne_icalabel.iclabel import iclabel_label_components

directory = testing.data_path() / "MEG" / "sample"
raw = read_raw(directory / "sample_audvis_trunc_raw.fif", preload=False)
raw.pick_types(eeg=True, exclude=[])
raw.load_data()
# preprocess
with raw.info._unlock():  # fake filtering, testing dataset is filtered between [0.1, 80] Hz
    raw.info["highpass"] = 1.
    raw.info["lowpass"] = 100.
raw.set_eeg_reference("average")


@pytest.mark.parametrize(
    "inst, exclude",
    (
        (raw, "bads"),
        (raw.copy().crop(0, 8), "bads"),
        (raw.copy().crop(0, 1), "bads"),
        (make_fixed_length_epochs(raw, duration=0.5, preload=True), "bads"),
        (make_fixed_length_epochs(raw, duration=1, preload=True), "bads"),
        (make_fixed_length_epochs(raw, duration=5, preload=True), "bads"),
        (raw, []),
        (raw.copy().crop(0, 8), []),
        (raw.copy().crop(0, 1), []),
        (make_fixed_length_epochs(raw, duration=0.5, preload=True), []),
        (make_fixed_length_epochs(raw, duration=1, preload=True), []),
        (make_fixed_length_epochs(raw, duration=5, preload=True), []),
    ),
)
def test_label_components(inst, exclude):
    """Check that label_components does not raise on various data shapes."""
    picks = pick_types(raw.info, eeg=True, exclude=exclude)
    ica = ICA(n_components=5, method="picard", fit_params=dict(ortho=False, extended=True))
    ica.fit(inst, picks=picks)
    labels = iclabel_label_components(inst, ica)
    assert labels.shape == (ica.n_components_, 7)


def test_warnings():
    """Test warnings issued when the raw|epochs|ica instance are not using the
    same algorithm/reference/filters as ICLabel."""
    times = np.linspace(0, 5, 2000)
    signals = np.array([np.sin(2 * np.pi * k * times) for k in (7, 22, 37)])
    coeffs = np.random.rand(6, 3)
    data = np.dot(coeffs, signals) + np.random.normal(0, 0.1, (coeffs.shape[0], times.size))

    raw = RawArray(
        data, create_info(["Fpz", "Cz", "CPz", "Oz", "M1", "M2"], sfreq=400, ch_types="eeg")
    )
    raw.set_montage("standard_1020")
    with raw.info._unlock():
        raw.info["highpass"] = 1.0

    # wrong raw, correct ica
    ica = ICA(n_components=3, method="infomax", fit_params=dict(extended=True), random_state=101)
    ica.fit(raw)
    with pytest.warns(RuntimeWarning, match="common average reference"):
        iclabel_label_components(raw, ica)
    with pytest.warns(RuntimeWarning, match="not filtered between 1 and 100 Hz"):
        iclabel_label_components(raw, ica)

    with raw.info._unlock():
        raw.info["lowpass"] = 100.0
    raw.set_eeg_reference("average")
    # infomax
    ica = ICA(n_components=3, method="infomax", fit_params=dict(extended=False), random_state=101)
    ica.fit(raw)
    with pytest.warns(RuntimeWarning, match="designed with extended infomax ICA"):
        iclabel_label_components(raw, ica)
    # fastica
    ica = ICA(n_components=3, fit_params=dict(tol=1e-2), random_state=101)
    ica.fit(raw)
    with pytest.warns(RuntimeWarning, match="designed with extended infomax ICA"):
        iclabel_label_components(raw, ica)
    # picard
    ica = ICA(n_components=3, method="picard", random_state=101)
    ica.fit(raw)
    with pytest.warns(RuntimeWarning, match="designed with extended infomax ICA"):
        iclabel_label_components(raw, ica)
