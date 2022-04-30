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
# fit ICA
ica = ICA(n_components=15, method="picard")
ica.fit(raw)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_label_components():
    """Simple test to check that label_components runs without raising."""
    labels = label_components(raw, ica, method="iclabel")
    assert labels is not None


def test_label_components_with_wrong_arguments():
    """Test that wrong arguments raise."""
    with pytest.raises(ValueError, match="Invalid value for the 'method' parameter"):
        label_components(raw, ica, method="101")
