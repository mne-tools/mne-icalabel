try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from mne.io import read_raw
from mne.io.eeglab.eeglab import _check_load_mat
from mne.preprocessing import read_ica_eeglab
import numpy as np
from scipy.io import loadmat

from mne_icalabel.features import retrieve_eeglab_icawinv, compute_ica_activations


# Raw/Epochs files with ICA decomposition
raw_eeglab_path = str(files("mne_icalabel.tests").joinpath("data/sample-raw.set"))
epo_eeglab_path = str(files("mne_icalabel.tests").joinpath("data/sample-epo.set"))

# ICA activation matrix for raw/epochs
raw_icaact_eeglab_path = str(
    files("mne_icalabel.tests").joinpath("data/icaact-raw.mat")
)
epo_icaact_eeglab_path = str(
    files("mne_icalabel.tests").joinpath("data/icaact-epo.mat")
)


def test_retrieve_eeglab_icawinv():
    """Test that the icawinv is correctly retrieved from an MNE ICA object."""
    # Raw instance
    ica = read_ica_eeglab(raw_eeglab_path)
    icawinv, _ = retrieve_eeglab_icawinv(ica)

    eeg = _check_load_mat(raw_eeglab_path, None)
    assert np.allclose(icawinv, eeg.icawinv)

    # Epoch instance


def test_compute_ica_activations():
    """Test that the icaact is correctly retrieved from an MNE ICA object."""
    # Raw instance
    raw = read_raw(raw_eeglab_path)
    ica = read_ica_eeglab(raw_eeglab_path)
    icaact = compute_ica_activations(raw, ica)

    icaact_eeglab = loadmat(raw_icaact_eeglab_path)["icaact"]
    assert np.allclose(icaact, icaact_eeglab, rtol=1e-8, atol=1e-4)

    # Epoch instance
