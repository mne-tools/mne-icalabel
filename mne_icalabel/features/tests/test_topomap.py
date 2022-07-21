import numpy as np
import pytest
from mne.datasets import testing
from mne.io import read_raw
from mne.preprocessing import ICA
from mne.utils import requires_version

from mne_icalabel.features import get_topomap, get_topomaps

directory = testing.data_path() / "MEG" / "sample"
raw = read_raw(directory / "sample_audvis_trunc_raw.fif", preload=False)
raw.pick_types(eeg=True)
raw.load_data()
ica = ICA(n_components=5, method="picard")
ica.fit(raw)


@requires_version("mne", "1.1")
def test_topomap_defaults():
    """Test scalp topography array generation"""
    topomaps = get_topomaps(ica, picks=None)
    assert isinstance(topomaps, np.ndarray)
    assert topomaps.shape == (ica.n_components_, 64, 64)

    # test single topo with fake data
    data = np.random.randint(1, 10, len(raw.ch_names))
    topomap = get_topomap(data, raw.info)
    assert isinstance(topomap, np.ndarray)
    assert topomap.shape == (64, 64)


@requires_version("mne", "1.1")
@pytest.mark.parametrize(
    "picks, res", [(0, 32), ([0, 1, 2], 50), (slice(1, 3), 128), (np.array([1, 2]), 10)]
)
def test_topomap_arguments(picks, res):
    """Test arguments that influence the output shape."""
    topomaps = get_topomaps(ica, picks=picks, res=res)
    assert isinstance(topomaps, np.ndarray)
    if isinstance(picks, int):
        n_components = 1
    elif isinstance(picks, slice):
        n_components = len(range(*picks.indices(ica.n_components_)))
    else:
        n_components = len(picks)
    assert topomaps.shape == (n_components, res, res)


@requires_version("mne", "1.1")
def test_invalid_arguments():
    """Test invalid arguments."""
    with pytest.raises(IndexError):
        get_topomaps(ica, picks="eeg")
