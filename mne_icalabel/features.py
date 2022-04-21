from mne.io import BaseRaw
from mne.preprocessing import ICA
import numpy as np
from numpy.typing import ArrayLike


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


def compute_ica_activations(
        raw: BaseRaw,
        ica: ICA
        ) -> ArrayLike:
    """Compute the ICA activations 'icaact' variable from an MNE ICA instance.

    Parameters
    ----------
    raw : Raw
        MNE Raw instance with data array in Volts.
    ica : ICA
        MNE ICA decomposition.

    Returns
    -------
    icaact : array
    """
    icawinv, weights = retrieve_eeglab_icawinv(ica)
    icasphere = np.eye(icawinv.shape[0])
    raw_data = raw.get_data(picks=ica.ch_names)
    icaact = (weights[0:ica.n_components_, :] @ icasphere) @ raw_data
    return icaact * 1e6
