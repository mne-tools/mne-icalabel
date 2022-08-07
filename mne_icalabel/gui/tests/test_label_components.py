import pytest
from mne.datasets import testing
from mne.io import BaseRaw, read_raw
from mne.preprocessing import ICA

from mne_icalabel.gui import label_ica_components

directory = testing.data_path() / "MEG" / "sample"
raw = read_raw(directory / "sample_audvis_trunc_raw.fif", preload=False)
raw.pick_types(eeg=True, exclude="bads")
raw.load_data()
# preprocess
with raw.info._unlock():  # fake filtering, testing dataset is filtered between [0.1, 80] Hz
    raw.info["highpass"] = 1.0
    raw.info["lowpass"] = 100.0
ica = ICA(n_components=5, random_state=12345, fit_params=dict(tol=1e-1))
ica.fit(raw)


def test_label_components_gui_display():
    ica_ = ica.copy()
    gui = label_ica_components(raw, ica_, show=False)
    # test setting the label
    assert isinstance(gui.inst, BaseRaw)
    assert isinstance(gui.ica, ICA)
    assert gui.n_components_ == ica.n_components_
    # the initial component should be 0
    assert gui.selected_component == 0


def test_invalid_arguments():
    """Test error error raised with invalid arguments."""
    with pytest.raises(TypeError, match="ica must be an instance of ICA"):
        label_ica_components(raw, 101)

    with pytest.raises(TypeError, match="inst must be an instance of raw or epochs"):
        label_ica_components(101, ica)

    with pytest.raises(ValueError, match="ICA instance should be fitted"):
        label_ica_components(raw, ICA(n_components=10, random_state=12345))
