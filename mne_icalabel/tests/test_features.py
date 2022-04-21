try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from mne.io.eeglab.eeglab import _check_load_mat
from mne.preprocessing import read_ica_eeglab
import numpy as np

from mne_icalabel.features import retrieve_eeglab_icawinv


# Raw files with ICA decomposition
raw_eeglab_path = str(files("mne_icalabel.tests").joinpath("data/sample.set"))


def test_retrieve_eeglab_icawinv():
    """Test that the icawinv is correctly retrieved from an MNE ICA object."""
    ica = read_ica_eeglab(raw_eeglab_path)
    icawinv = retrieve_eeglab_icawinv(ica)

    eeg = _check_load_mat(raw_eeglab_path, None)
    assert np.allclose(icawinv, eeg.icawinv)
