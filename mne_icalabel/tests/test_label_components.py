import pytest
from mne.datasets import sample
from mne.io import read_raw
from mne.preprocessing import ICA

from mne_icalabel import label_components

directory = sample.data_path() / "MEG" / "sample"
raw = read_raw(directory / "sample_audvis_raw.fif", preload=False)
raw.crop(0, 10).pick_types(eeg=True, exclude="bads")
raw.load_data()
# preprocess
raw.filter(l_freq=1.0, h_freq=100.0)
raw.set_eeg_reference("average")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("n_components", (5, 15))
def test_label_components(n_components):
    """Simple test to check that label_components runs without raising."""
    ica = ICA(n_components=n_components, method="picard")
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
