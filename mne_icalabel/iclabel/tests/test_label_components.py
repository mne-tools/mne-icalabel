import numpy as np
import pytest
from mne import create_info
from mne.datasets import sample
from mne.io import RawArray, read_raw
from mne.preprocessing import ICA

from mne_icalabel.iclabel import iclabel_label_components

directory = sample.data_path() / "MEG" / "sample"
raw = read_raw(directory / "sample_audvis_raw.fif", preload=False)
raw.crop(0, 10).pick_types(eeg=True, exclude="bads")
raw.load_data()
# preprocess
raw.filter(l_freq=1.0, h_freq=100.0)
raw.set_eeg_reference("average")
# fit ICA
ica = ICA(n_components=5, method="picard", random_state=101)
ica.fit(raw)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_label_components():
    """Simple test to check that label_components runs without raising."""
    labels = iclabel_label_components(raw, ica)
    assert labels.shape == (ica.n_components_, 7)


def test_warnings():
    """Test warnings issued when the raw|epochs|ica instance are not using the
    same algorithm/reference/filters as ICLabel."""
    data = np.random.randint(low=1, high=10, size=(6, 10000)) / 1000
    raw = RawArray(
        data, create_info(["Fpz", "CPz", "Oz", "Fp1", "Fp2", "Cz"], sfreq=500, ch_types="eeg")
    )
    raw.set_montage("standard_1020")
    raw.filter(1.0, None)

    # wrong raw, correct ica
    ica = ICA(n_components=4, method="infomax", fit_params=dict(extended=True), random_state=101)
    ica.fit(raw)
    with pytest.warns(RuntimeWarning, match="common average reference"):
        iclabel_label_components(raw, ica)
    with pytest.warns(RuntimeWarning, match="not filtered between 1 and 100 Hz"):
        iclabel_label_components(raw, ica)

    raw.filter(1.0, 100.0)
    raw.set_eeg_reference("average")
    # infomax
    ica = ICA(n_components=4, method="infomax", fit_params=dict(extended=False), random_state=101)
    ica.fit(raw)
    with pytest.warns(RuntimeWarning, match="designed with extended infomax ICA"):
        iclabel_label_components(raw, ica)
    # fastica
    ica = ICA(n_components=4, fit_params=dict(tol=1e-2), random_state=101)
    ica.fit(raw)
    with pytest.warns(RuntimeWarning, match="designed with extended infomax ICA"):
        iclabel_label_components(raw, ica)
    # picard
    ica = ICA(n_components=4, method="picard", random_state=101)
    ica.fit(raw)
    with pytest.warns(RuntimeWarning, match="designed with extended infomax ICA"):
        iclabel_label_components(raw, ica)
