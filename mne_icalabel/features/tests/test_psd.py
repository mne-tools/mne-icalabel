import numpy as np
from mne.datasets import testing
from mne.io import read_raw
from mne.preprocessing import ICA

from mne_icalabel.features import get_psds

directory = testing.data_path() / "MEG" / "sample"
raw = read_raw(directory / "sample_audvis_trunc_raw.fif", preload=False)
raw.pick_types(eeg=True)
raw.load_data()
ica = ICA(n_components=7, method="picard")
ica.fit(raw)


def test_psd_feature():
    """Test scalp topography array generation"""
    psds_mean = get_psds(ica, raw)
    assert isinstance(psds_mean, np.ndarray)
