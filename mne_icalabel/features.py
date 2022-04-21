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
    """
    n_components = ica.n_components_
    s = np.sqrt(ica.pca_explained_variance_)[:n_components]
    u = ica.unmixing_matrix_ / s
    v = ica.pca_components_[:n_components, :]
    return np.linalg.pinv((u * s) @ v)
