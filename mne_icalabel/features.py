from typing import Union

from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA
import numpy as np
from numpy.typing import ArrayLike


def get_features(inst: Union[BaseRaw, BaseEpochs], ica: ICA):
    """
    Generates the features for ICLabel neural network.

    Parameters
    ----------
    inst : Raw | Epoch
        MNE Raw/Epoch instance with data array in Volts.
    ica : ICA
        MNE ICA decomposition.
    """
    icawinv, _ = retrieve_eeglab_icawinv(ica)
    icaact = compute_ica_activations(inst, ica)

    # compute topographic feature (float32)
    psd = eeg_topoplot()

    # compute psd feature (float32)
    psd = eeg_rpsd()

    # compute autocorr feature (float32)
    if isinstance(inst, BaseRaw):
        if 5 < inst.times.size / inst.info['sfreq']:
            autocorr = eeg_autocorr_welch()
        else:
            autocorr = eeg_autocorr()
    else:
        autocorr = eeg_autocorr_fftw()


def retrieve_eeglab_icawinv(
    ica: ICA,
) -> ArrayLike:
    """
    Retrieves 'icawinv' from an MNE ICA instance.

    Parameters
    ----------
    ica : ICA
        MNE ICA decomposition.

    Returns
    -------
    icawinv : array
    weights : array
    """
    n_components = ica.n_components_
    s = np.sqrt(ica.pca_explained_variance_)[:n_components]
    u = ica.unmixing_matrix_ / s
    v = ica.pca_components_[:n_components, :]
    weights = (u * s) @ v
    return np.linalg.pinv(weights), weights


def compute_ica_activations(inst: Union[BaseRaw, BaseEpochs], ica: ICA) -> ArrayLike:
    """Compute the ICA activations 'icaact' variable from an MNE ICA instance.

    Parameters
    ----------
    inst : Raw | Epoch
        MNE Raw/Epoch instance with data array in Volts.
    ica : ICA
        MNE ICA decomposition.

    Returns
    -------
    icaact : array
        raw: (n_components, n_samples)
        epoch: (n_components, n_samples, n_trials)
    """
    icawinv, weights = retrieve_eeglab_icawinv(ica)
    icasphere = np.eye(icawinv.shape[0])
    data = inst.get_data(picks=ica.ch_names) * 1e6
    icaact = (weights[0 : ica.n_components_, :] @ icasphere) @ data
    # move trial (epoch) dimension to the end
    if icaact.ndim == 3:
        assert isinstance(inst, BaseEpochs)  # sanity-check
        icaact = icaact.transpose([1, 2, 0])
    return icaact


# ----------------------------------------------------------------------------
def eeg_topoplot():
    """Topoplot feature."""
    pass


# ----------------------------------------------------------------------------
def eeg_rpsd():
    """PSD feature."""
    pass


# ----------------------------------------------------------------------------
def eeg_autocorr_welch():
    """Autocorrelation feature applied on raw object with at least 5 * fs
    samples (5 seconds)."""
    pass


def eeg_autocorr():
    """Autocorrelation feature applied on raw object that do not have enough
    sampes for eeg_autocorr_welch."""
    pass


def eeg_autocorr_fftw():
    """Autocorrelation feature applied on epoch object."""
    pass
