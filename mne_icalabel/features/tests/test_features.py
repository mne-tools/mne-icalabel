from mne.datasets import testing
from mne.io import read_raw
from mne.preprocessing import ICA
from numpy.typing import NDArray
from mne_icalabel.features import topomap

def _create_test_inst_and_ica():
    directory = testing.data_path() / "MEG" / "sample"
    raw = read_raw(directory / "sample_audvis_trunc_raw.fif", preload=False)
    raw.pick_types(eeg=True)
    raw.load_data()
    ica = ICA(n_components=5, method="picard")
    ica.fit(raw)
    return ica, raw

def test_topomap_feature():
    """Test scalp topography array generation"""
    ica, inst = _create_test_inst_and_ica()
    topo_array = topomap.get_topo_array(ica, picks="eeg")
    assert isinstance(topo_array, NDArray)
    assert topo_array.shape == (ica.n_components_, 64, 64)
