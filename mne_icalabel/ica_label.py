import numpy as np
from numpy.typing import ArrayLike
import mne

from .ica_features import rpsd, autocorr_fftw, topoplot, mne_to_eeglab_locs


def ica_eeg_features(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    subset: np.array = None,
    pct_data: int = 100,
) -> ArrayLike:
    """
    Generates the feature nd-array for ICLabel.

    Parameters
    ----------
    raw : instance of Raw
        The Raw object that ICA is applied to.
    ica : instance of ICA
        The ICA instance that was fitted to ``raw``.
    subset : np.ndarray
        A subset to take on the RPSD.
        TODO: Not sure what this input argument is for.
    pct_data (int, optional):
        Defaults to 100.

    Returns:
        np.ndarray: Feature matrix (4D)
    """
    n_components = ica.n_components_
    sfreq = int(raw.info["sfreq"])
    # pnts = icaact.shape[1]

    # get the weights * sphere and compute ica weight inverse
    s = np.sqrt(ica.pca_explained_variance_)[:n_components]
    u = ica.unmixing_matrix_ / s
    v = ica.pca_components_[:n_components, :]
    ica_weights = (u * s) @ v
    icawinv = np.linalg.pinv(ica_weights)

    # ica sphere - assumed to be identity
    icasphere = np.eye(icawinv.shape[0])

    # get the ica activations
    raw_data = raw.get_data(picks=ica.ch_names)  # * 1e6
    icaact = (ica_weights[0:n_components, :] @ icasphere) @ raw_data

    # make sure ICA activation is 3D to account for "trial" dimension
    if icaact.ndim == 2:
        icaact = np.expand_dims(icaact, axis=2)

    # get the polar coordinates of electrode locations
    rd, th = mne_to_eeglab_locs(raw)

    # Generate topoplot features
    topo = np.zeros((32, 32, 1, n_components))
    plotchans = np.squeeze(np.argwhere(~np.isnan(np.squeeze(th))))
    print(n_components)
    for it in range(n_components):
        temp_topo = topoplot(
            icawinv=icawinv[:, it], theta_coords=th,
            rho_coords=rd, picks=plotchans
        )
        np.nan_to_num(temp_topo, copy=False)  # Set NaN values to 0 in-place
        topo[:, :, 0, it] = temp_topo / np.max(np.abs(temp_topo))

    # Generate PSD Features
    # RPSD is sensitive to sfreq as a float
    psd = rpsd(icaact=icaact, sfreq=sfreq, pct_data=pct_data, subset=subset)

    # Autocorrelation
    autocorr = autocorr_fftw(icaact=icaact, sfreq=sfreq)
    autocorr = np.expand_dims(autocorr, (2, 3))
    autocorr = np.transpose(autocorr, [2, 1, 3, 0])

    return [0.99 * topo, 0.99 * psd, 0.99 * autocorr]
