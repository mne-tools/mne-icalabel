import numpy as np
from mne.datasets import testing
from mne.io import read_raw
from mne.preprocessing import ICA

from mne_icalabel.features import get_topomap, get_topomaps

directory = testing.data_path() / "MEG" / "sample"
raw = read_raw(directory / "sample_audvis_trunc_raw.fif", preload=False)
raw.pick_types(eeg=True)
raw.load_data()
ica = ICA(n_components=5, method="picard")
ica.fit(raw)


def test_topomap_defaults():
    """Test scalp topography array generation"""
    topomaps = get_topomaps(ica, picks=None)
    assert isinstance(topomaps, np.ndarray)
    assert topomaps.shape == (ica.n_components_, 64, 64)
    # Need to think of an advanced test because this will fail when 'picks' != None
    # then the shape of topomap will be ( len(picks), 64,64)

    # test single topo with fake data
    data = np.random.randint(1, 10, len(raw.ch_names))
    topomap = get_topomap(data, raw.info)
    assert isinstance(topomap, np.ndarray)
    assert topomap.shape == (64, 64)
