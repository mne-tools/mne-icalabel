import re

import numpy as np
import pytest
from mne import pick_info
from mne.datasets import testing
from mne.io import read_raw
from mne.preprocessing import ICA
from mne.utils import check_version

if check_version("mne", "1.6"):
    from mne._fiff.pick import _pick_data_channels, _picks_to_idx
else:
    from mne.io.pick import _pick_data_channels, _picks_to_idx

from mne_icalabel.features import get_topomaps

directory = testing.data_path() / "MEG" / "sample"
raw = read_raw(directory / "sample_audvis_trunc_raw.fif", preload=False)
raw.pick_types(eeg=True, meg=True)
raw.load_data()
ica = ICA(n_components=5, method="picard")
ica.fit(raw)
ica_eeg = ICA(n_components=5, method="picard")
ica_eeg.fit(raw.pick_types(eeg=True))


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.parametrize("ica", (ica, ica_eeg))
def test_topomap_defaults(ica):
    """Test scalp topography array generation"""
    topomaps = get_topomaps(ica, picks=None)
    assert isinstance(topomaps, dict)
    for topomaps_ in topomaps.values():
        assert isinstance(topomaps_, np.ndarray)
        assert topomaps_.shape == (ica.n_components_, 64, 64)
    ch_picks = _pick_data_channels(ica.info, exclude=())
    ch_types = pick_info(ica.info, ch_picks).get_channel_types(unique=True)
    assert sorted(topomaps) == sorted(ch_types)


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.parametrize(
    "picks, res", [(0, 32), ([0, 1, 2], 50), (slice(1, 3), 128), (np.array([1, 2]), 10)]
)
def test_topomap_arguments(picks, res):
    """Test arguments that influence the output shape."""
    topomaps = get_topomaps(ica, picks=picks, res=res)
    assert isinstance(topomaps, dict)
    ic_picks = _picks_to_idx(ica.n_components_, picks)
    for topomaps_ in topomaps.values():
        assert isinstance(topomaps_, np.ndarray)
        assert topomaps_.shape == (ic_picks.size, res, res)


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_interpolation_arguments():
    """Test arguments that influence the interpolation."""
    # default
    topomaps1 = get_topomaps(ica)

    # image_interp
    topomaps2 = get_topomaps(ica, image_interp="linear")
    topomaps3 = get_topomaps(ica, image_interp="nearest")
    for topomaps in (topomaps2, topomaps3):
        assert sorted(topomaps1) == sorted(topomaps)
        for ch_type in topomaps1:
            assert not np.allclose(topomaps1[ch_type], topomaps[ch_type])
    del topomaps2, topomaps3, topomaps

    # border
    topomaps4 = get_topomaps(ica, border=5)
    assert sorted(topomaps1) == sorted(topomaps4)
    for ch_type in topomaps1:
        assert not np.allclose(topomaps1[ch_type], topomaps4[ch_type])
    del topomaps4

    # extrapolate
    topomaps5 = get_topomaps(ica, extrapolate="box")
    topomaps6 = get_topomaps(ica, extrapolate="head")
    topomaps7 = get_topomaps(ica, extrapolate="local")
    for topomaps in (topomaps5, topomaps6, topomaps7):
        assert sorted(topomaps1) == sorted(topomaps)
    del topomaps
    for ch_type in topomaps1:
        assert not np.allclose(topomaps1[ch_type], topomaps5[ch_type])
        if ch_type == "eeg":
            assert np.allclose(topomaps1[ch_type], topomaps6[ch_type])
        else:
            assert not np.allclose(topomaps1[ch_type], topomaps6[ch_type])
        if ch_type in ("grad", "mag"):
            assert np.allclose(topomaps1[ch_type], topomaps7[ch_type])
        else:
            assert not np.allclose(topomaps1[ch_type], topomaps7[ch_type])
    del topomaps5, topomaps6, topomaps7
    del topomaps1


def test_invalid_arguments():
    """Test invalid arguments."""
    with pytest.raises(TypeError, match="ica must be an instance of ICA"):
        get_topomaps(101)
    with pytest.raises(RuntimeError, match="The provided ICA instance was not fitted."):
        get_topomaps(ICA(n_components=5, method="picard"))

    with pytest.raises(
        TypeError,
        match=re.escape(
            "picks must be a list of int (indices) or list of str (names)."
        ),
    ):
        get_topomaps(ica, picks=101 + 101j)
    with pytest.raises(TypeError, match="Strings are not supported."):
        get_topomaps(ica, picks="101")
    with pytest.raises(ValueError, match="All picks must be < n_channels"):
        get_topomaps(ica, picks=6)

    with pytest.raises(TypeError, match="res must be an int"):
        get_topomaps(ica, res="101")
    with pytest.raises(TypeError, match="res must be an int"):
        get_topomaps(ica, res=True)
    with pytest.raises(ValueError, match="strictly positive integer. Provided '-101'"):
        get_topomaps(ica, res=-101)
