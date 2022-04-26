try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from mne import read_epochs_eeglab
from mne.io import read_raw
import numpy as np
from scipy.io import loadmat
import pytest

from mne_icalabel.utils import mne_to_eeglab_locs, gdatav4


# Raw/Epochs files with ICA decomposition
raw_eeglab_path = str(
    files("mne_icalabel.tests").joinpath("data/datasets/sample-raw.set")
)
epo_eeglab_path = str(
    files("mne_icalabel.tests").joinpath("data/datasets/sample-epo.set")
)


# Electrode locations
loc_raw_path = str(files("mne_icalabel.tests").joinpath("data/utils/loc-raw.mat"))
loc_epo_path = str(files("mne_icalabel.tests").joinpath("data/utils/loc-raw.mat"))

# Grid data interpolation
gdatav4_raw_path = str(
    files("mne_icalabel.tests").joinpath("data/utils/gdatav4-raw.mat")
)
gdatav4_epo_path = str(
    files("mne_icalabel.tests").joinpath("data/utils/gdatav4-epo.mat")
)


def test_loc():
    """Test conversion of MNE montage to EEGLAB loc.

    This test works because MNE does the conversion from EEGLAB to MNE montage
    when loading the datasets."""
    # from raw
    raw = read_raw(raw_eeglab_path, preload=True)
    rd, th = mne_to_eeglab_locs(raw)
    eeglab_loc = loadmat(loc_raw_path)["loc"][0, 0]
    eeglab_rd = eeglab_loc["rd"]
    eeglab_th = eeglab_loc["th"]
    assert np.allclose(rd, eeglab_rd, atol=1e-8)
    assert np.allclose(th, eeglab_th, atol=1e-8)

    # from epochs
    epochs = read_epochs_eeglab(epo_eeglab_path)
    rd, th = mne_to_eeglab_locs(epochs)
    eeglab_loc = loadmat(loc_epo_path)["loc"][0, 0]
    eeglab_rd = eeglab_loc["rd"]
    eeglab_th = eeglab_loc["th"]
    assert np.allclose(rd, eeglab_rd, atol=1e-8)
    assert np.allclose(th, eeglab_th, atol=1e-8)


# TODO: Warnings should be fixed at some point.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("file", (gdatav4_raw_path, gdatav4_epo_path))
def test_gdatav4(file):
    """Test grid data interpolation."""
    # ------------------------- Test without meshgrid ------------------------
    # load inputs from MATLAB
    eeglab_gdata = loadmat(file)["gdatav4"][0, 0]
    eeglab_inty = eeglab_gdata["inty"]
    eeglab_intx = eeglab_gdata["intx"]
    eeglab_intValues = eeglab_gdata["intValues"]
    eeglab_yi = eeglab_gdata["yi"]
    eeglab_xi = eeglab_gdata["xi"]

    # compute output in Python
    Xi, Yi, Zi = gdatav4(
        eeglab_intx, eeglab_inty, eeglab_intValues, eeglab_xi, eeglab_yi
    )

    # load outputs from MATLAB
    eeglab_Xi = eeglab_gdata["Xi"]
    eeglab_Yi = eeglab_gdata["Yi"]
    eeglab_Zi = eeglab_gdata["Zi"]

    # output in EEGLAB repeats the same vector to form a square matrix.
    # For Xi -> the row is repeated
    # For Yi -> the column is repeated
    # For Zi -> python outputs the diagonal
    assert np.allclose(eeglab_Xi, np.tile(Xi, (Xi.size, 1)), atol=1e-8)
    assert np.allclose(eeglab_Yi, np.tile(Yi.T, Yi.size), atol=1e-8)
    assert np.allclose(np.diagonal(eeglab_Zi), Zi, atol=1e-8)

    # --------------------------- Test with meshgrid -------------------------
    # create mesh
    xq, yq = np.meshgrid(eeglab_xi, eeglab_yi)

    # compute output in Python
    Xi, Yi, Zi = gdatav4(eeglab_intx, eeglab_inty, eeglab_intValues, xq, yq)

    # compare
    assert np.allclose(Xi, eeglab_Xi, atol=1e-8)
    assert np.allclose(Yi, eeglab_Yi, atol=1e-8)
    # Zi has to be transposed in Python
    assert np.allclose(Zi.T, eeglab_Zi, atol=1e-8)
