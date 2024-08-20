import numpy as np
import pytest
from mne import create_info, make_fixed_length_epochs, pick_types
from mne.datasets import testing
from mne.io import RawArray, read_raw
from mne.preprocessing import ICA

from mne_icalabel.config import ICA_LABELS_TO_MNE
from mne_icalabel.iclabel import iclabel_label_components
from mne_icalabel.utils._tests import requires_module

directory = testing.data_path() / "MEG" / "sample"
raw = read_raw(directory / "sample_audvis_trunc_raw.fif", preload=False)
raw.pick_types(eeg=True, exclude=[])
raw.load_data()
# preprocess, fake filtering, testing dataset is filtered between [0.1, 80] Hz
with raw.info._unlock():
    raw.info["highpass"] = 1.0
    raw.info["lowpass"] = 100.0
raw.set_eeg_reference("average")


@pytest.mark.parametrize(
    ("inst", "exclude"),
    [
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
    ],
)
@requires_module("onnxruntime")
def test_label_components_onnx(inst, exclude):
    """Check that label_components does not raise on various data shapes."""
    picks = pick_types(inst.info, eeg=True, exclude=exclude)
    ica = ICA(
        n_components=5, method="picard", fit_params=dict(ortho=False, extended=True)
    )
    ica.fit(inst, picks=picks)
    labels = iclabel_label_components(inst, ica, inplace=False, backend="onnx")
    assert labels.shape == (ica.n_components_, 7)
    assert len(ica.labels_) == 0
    labels2 = iclabel_label_components(inst, ica, inplace=True, backend="onnx")
    assert sorted(ica.labels_.keys()) == sorted(ICA_LABELS_TO_MNE.values())
    assert np.allclose(labels, labels2)


@pytest.mark.parametrize(
    ("inst", "exclude"),
    [
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
    ],
)
@requires_module("torch")
def test_label_components_torch(inst, exclude):
    """Check that label_components does not raise on various data shapes."""
    picks = pick_types(inst.info, eeg=True, exclude=exclude)
    ica = ICA(
        n_components=5, method="picard", fit_params=dict(ortho=False, extended=True)
    )
    ica.fit(inst, picks=picks)
    labels = iclabel_label_components(inst, ica, inplace=False, backend=None)
    assert labels.shape == (ica.n_components_, 7)
    assert len(ica.labels_) == 0
    labels2 = iclabel_label_components(inst, ica, inplace=True, backend=None)
    assert sorted(ica.labels_.keys()) == sorted(ICA_LABELS_TO_MNE.values())
    assert np.allclose(labels, labels2)

    picks = pick_types(inst.info, eeg=True, exclude=exclude)
    ica = ICA(
        n_components=5, method="picard", fit_params=dict(ortho=False, extended=True)
    )
    ica.fit(inst, picks=picks)
    labels = iclabel_label_components(inst, ica, inplace=False, backend="torch")
    assert labels.shape == (ica.n_components_, 7)
    assert len(ica.labels_) == 0
    labels2 = iclabel_label_components(inst, ica, inplace=True, backend="torch")
    assert sorted(ica.labels_.keys()) == sorted(ICA_LABELS_TO_MNE.values())
    assert np.allclose(labels, labels2)


def test_warnings(rng):
    """Test warnings issued when the instance does not meet the requirements."""
    times = np.linspace(0, 5, 2000)
    signals = np.array([np.sin(2 * np.pi * k * times) for k in (7, 22, 37)])
    coeffs = rng.random((6, 3))
    data = np.dot(coeffs, signals) + rng.normal(0, 0.1, (coeffs.shape[0], times.size))

    raw = RawArray(
        data,
        create_info(["Fpz", "Cz", "CPz", "Oz", "M1", "M2"], sfreq=400, ch_types="eeg"),
    )
    raw.set_montage("standard_1020")
    with raw.info._unlock():
        raw.info["highpass"] = 1.0

    # wrong raw, correct ica
    ica = ICA(
        n_components=3,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=101,
    )
    ica.fit(raw)
    with (
        pytest.warns(RuntimeWarning, match="common average reference"),
        pytest.warns(RuntimeWarning, match="not filtered between 1 and 100 Hz"),
    ):
        iclabel_label_components(raw, ica)
    with raw.info._unlock():
        raw.info["lowpass"] = 100.0
    raw.set_eeg_reference("average")

    # infomax
    ica = ICA(
        n_components=3,
        method="infomax",
        fit_params=dict(extended=False),
        random_state=101,
    )
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

    raw = read_raw(directory / "sample_audvis_trunc_raw.fif", preload=False)
    raw.pick_types(meg=True)
    raw.load_data()
    ica = ICA(
        n_components=3,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=101,
    )
    ica.fit(raw)
    # raw without EEG channels
    with pytest.raises(RuntimeError, match="Could not find EEG channels"):
        iclabel_label_components(raw, ica)

    epochs = make_fixed_length_epochs(raw)
    ica = ICA(
        n_components=3,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=101,
    )
    ica.fit(epochs)
    # epochs without EEG channels
    with pytest.raises(RuntimeError, match="Could not find EEG channels"):
        iclabel_label_components(epochs, ica)


def test_comp_in_labels_():
    """Test that components already in labels_ are not added again."""
    picks = pick_types(raw.info, eeg=True, exclude="bads")
    ica = ICA(
        n_components=5,
        method="picard",
        fit_params=dict(ortho=False, extended=True),
        random_state=101,
    )
    ica.fit(raw, picks=picks)
    ica.labels_["brain"] = [3]
    ica.labels_["other"] = []
    iclabel_label_components(raw, ica, inplace=True)
    assert ica.labels_["brain"] == [3]
    assert ica.labels_["other"] == [1, 2, 4]
