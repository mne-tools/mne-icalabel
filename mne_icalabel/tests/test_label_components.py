import pytest
from mne.datasets import testing
from mne.io import read_raw
from mne.preprocessing import ICA

from mne_icalabel import label_components

directory = testing.data_path() / "MEG" / "sample"
raw = read_raw(directory / "sample_audvis_trunc_raw.fif", preload=False)
raw.pick_types(eeg=True, exclude="bads")
raw.load_data()
# preprocess
with raw.info._unlock():  # fake filtering, testing dataset is filtered between [0.1, 80] Hz
    raw.info["highpass"] = 1.0
    raw.info["lowpass"] = 100.0
raw.set_eeg_reference("average")


@pytest.mark.parametrize("n_components", (5, 15))
def test_label_components(n_components):
    """Simple test to check that label_components runs without raising."""
    ica = ICA(n_components=n_components, method="infomax", fit_params=dict(extended=True))
    ica.fit(raw)
    labels = label_components(raw, ica, method="iclabel")
    assert isinstance(labels, dict)
    assert labels["y_pred_proba"].ndim == 1
    assert labels["y_pred_proba"].shape[0] == ica.n_components_


def test_label_components_with_wrong_arguments():
    """Test that wrong arguments raise."""
    ica = ICA(n_components=3, method="picard")
    ica.fit(raw)
    with pytest.raises(ValueError, match="Invalid value for the 'method' parameter"):
        label_components(raw, ica, method="101")
