from pathlib import Path

import numpy as np
import pytest
from mne import read_epochs_eeglab
from mne.io import read_raw
from scipy.io import loadmat

from mne_icalabel.datasets import icalabel
from mne_icalabel.iclabel.utils import _gdatav4, _mne_to_eeglab_locs, _next_power_of_2

dataset_path = Path(icalabel.data_path()) / "iclabel"

# Raw/Epochs files with ICA decomposition
raw_eeglab_path = dataset_path / "datasets/sample-raw.set"
epo_eeglab_path = dataset_path / "datasets/sample-epo.set"

# Electrode locations
loc_raw_path = dataset_path / "utils/loc-raw.mat"
loc_epo_path = dataset_path / "utils/loc-raw.mat"

# Grid data interpolation
gdatav4_raw_path = dataset_path / "utils/gdatav4-raw.mat"
gdatav4_epo_path = dataset_path / "utils/gdatav4-epo.mat"


# General readers
reader = {"raw": read_raw, "epo": read_epochs_eeglab}
kwargs = {"raw": dict(preload=True), "epo": dict()}


@pytest.mark.parametrize(
    "file, eeglab_result_file",
    [(raw_eeglab_path, loc_raw_path), (epo_eeglab_path, loc_epo_path)],
)
def test_loc(file, eeglab_result_file):
    """Test conversion of MNE montage to EEGLAB loc.

    This test works because MNE does the conversion from EEGLAB to MNE montage
    when loading the datasets."""
    type_ = str(Path(file).stem)[-3:]
    inst = reader[type_](file, **kwargs[type_])
    rd, th = _mne_to_eeglab_locs(inst)
    eeglab_loc = loadmat(eeglab_result_file)["loc"][0, 0]
    eeglab_rd = eeglab_loc["rd"]
    eeglab_th = eeglab_loc["th"]
    assert np.allclose(rd, eeglab_rd, atol=1e-8)
    assert np.allclose(th, eeglab_th, atol=1e-8)


# TODO: Warnings should be fixed at some point.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("file", (gdatav4_raw_path, gdatav4_epo_path))
def test_gdatav4(file):
    """Test grid data interpolation."""
    # load inputs from MATLAB
    eeglab_gdata = loadmat(file)["gdatav4"][0, 0]
    eeglab_inty = eeglab_gdata["inty"]
    eeglab_intx = eeglab_gdata["intx"]
    eeglab_intValues = eeglab_gdata["intValues"]
    eeglab_yi = eeglab_gdata["yi"]
    eeglab_xi = eeglab_gdata["xi"]

    # create mesh
    xq, yq = np.meshgrid(eeglab_xi, eeglab_yi)

    # compute output in Python
    Xi, Yi, Zi = _gdatav4(eeglab_intx, eeglab_inty, eeglab_intValues, xq, yq)

    # load outputs from MATLAB
    eeglab_Xi = eeglab_gdata["Xi"]
    eeglab_Yi = eeglab_gdata["Yi"]
    eeglab_Zi = eeglab_gdata["Zi"]

    # compare
    assert np.allclose(Xi, eeglab_Xi, atol=1e-8)
    assert np.allclose(Yi, eeglab_Yi, atol=1e-8)
    # Zi has to be transposed in Python
    assert np.allclose(Zi.T, eeglab_Zi, atol=1e-8)


def test_next_power_of_2():
    """Test that next_power_of_2 works as intended."""
    x = [0, 10, 200, 400]
    expected = [1, 16, 256, 512]
    for k, exp in zip(x, expected):
        val = _next_power_of_2(k)
        assert exp == val
