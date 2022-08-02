import numpy as np
import pytest
from mne.datasets import testing
from mne.io import read_raw
from mne.preprocessing import ICA

from mne_icalabel.features import get_psds

directory = testing.data_path() / "MEG" / "sample"
inst = read_raw(directory / "sample_audvis_trunc_raw.fif", preload=False)
inst.pick_types(eeg=True)
inst.load_data()
ica = ICA(n_components=5, method="picard")
ica.fit(inst)


def test_psd_default():
    """Test scalp topography array generation."""
    psds = get_psds(ica, inst)
    assert isinstance(psds, np.ndarray)
    assert psds.shape[0] == ica.n_components_


def test_psd_nan():
    """Test handling of NaN in any channel."""
    inst_2 = inst.copy()
    inst_2[0, 100] = np.nan
    pytest.raises(ValueError, get_psds, ica, inst_2)


@pytest.mark.parametrize(
    "picks", [0, [0, 1, 2], slice(1, 3), np.array([1, 2])], "fmax", [[210, 410]]
)
def test_psd_arguments(picks):
    """Test arguments that influence the output shape."""
    psds = get_psds(ica, inst, picks=picks)
    assert isinstance(psds, np.ndarray)
    if isinstance(picks, int):
        n_components = 1
    elif isinstance(picks, slice):
        n_components = len(range(*picks.indices(ica.n_components_)))
    else:
        n_components = len(picks)
    assert psds.shape[0] == n_components


def test_invalid_arguments():
    """Test invalid arguments."""
    with pytest.raises(ValueError):
        get_psds(ica, inst, picks="eeg")
    with pytest.raises(ValueError, match="All picks must be < n_channels"):
        get_psds(ica, inst, picks=10)
    with pytest.raises(ValueError):
        get_psds(ica, inst, fmax=[210, 410])
    with pytest.raises(ValueError):
        get_psds(ica, inst, fmax="string")
    with pytest.raises(ValueError):
        get_psds(ica, inst, normalization="string")
