import numpy as np
import math
import scipy.signal
import warnings

from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA

from eeg_features import eeg_topoplot

from .ica_features import rpsd, autocorr_fftw, topoplot


def ica_label(inst, verbose=True, **ica_kwargs):
    """Automatically label ICA components and return a cleaned instance.

    Parameters
    ----------
    inst : instance of Raw, or Epochs
        MNE Raw, or Epochs object.
    verbose : bool, optional
        Verbosity, by default True.
    ica_kwargs : dict, optional
        ICA keyword arguments to be passed to `mne.preprocessing.ICA`.

    Returns
    -------
    clean_inst : instance of Raw, or Epochs
        MNE Raw, or Epochs object with ICA components that were
        classified as non-Brain activity removed.
    """
    if not isinstance(inst, BaseEpochs) and not isinstance(inst, BaseRaw):
        raise RuntimeError(
            f'{inst} should be a Raw or Epochs MNE-Python object.')

    # first prepare the data and remove drifts
    inst.load_data()
    filt_inst = inst.copy().filter(l_freq=1., h_freq=None, verbose=False)
    sfreq = inst.info['sfreq']

    # fit ICA
    ica = ICA(**ica_kwargs)
    ica.fit(filt_inst, verbose=verbose)

    # get the sources
    ica_activations = ica.get_sources(inst)
    ica_weights = ica.mixing_matrix_
    ica_sphere = ica.pca_components_

    # generate the feature matrix
    feature_matrix = eeg_features()

    # feed into model and label each component
    exclude = _call_model(feature_matrix)

    # remix
    clean_inst = ica.apply(inst, exclude=exclude)
    return clean_inst


def eeg_features(icaact: np.array,
                 trials: int,
                 srate: float,
                 pnts: int,
                 subset: np.array,
                 icaweights: np.array,
                 icawinv: np.array,
                 Th: np.array,
                 Rd: np.array,
                 plotchans: np.array,
                 pct_data: int = 100) -> np.array:
    """
    TODO: make this work with Raw
    Generates the feature nd-array for ICLabel.

    Args:
        icaact (np.array): ICA activation waveforms
        trials (int): Number of trials
        srate (float): Sampling Rate
        pnts (int): Number of Points
        icaweights (np.array): ICA Weights
        nfreqs (int): Number of frequencies
        icawinv (np.array): pinv(EEG.icaweights*EEG.icasphere)
        Th (np.array): Theta coordinates of electrodes (polar)
        Rd (np.array): Rho coordinates of electrodes (polar)
        plotchans (np.array): plot channels
        pct_data (int, optional): . Defaults to 100.

    Returns:
        np.array: Feature matrix (4D)
    """
    # Generate topoplot features
    ncomp = icawinv.shape[1]
    topo = np.zeros((32, 32, 1, ncomp))
    plotchans -= 1
    for it in range(ncomp):
        temp_topo = topoplot(
            icawinv=icawinv[:, it:it + 1], Th=Th, Rd=Rd, plotchans=plotchans)
        np.nan_to_num(temp_topo, copy=False)  # Set NaN values to 0 in-place
        topo[:, :, 0, it] = temp_topo / np.max(np.abs(temp_topo))

    # Generate PSD Features
    psd = rpsd(icaact=icaact, icaweights=icaweights, trials=trials,
               srate=srate, pnts=pnts, subset=subset)

    nfreq = psd.shape[1]
    if nfreq < 100:
        psd = np.concatenate(
            [psd, np.tile(psd[:, -2:-1], (1, 100 - nfreq))], axis=1)

    for linenoise_ind in [50, 60]:
        linenoise_around = np.array([linenoise_ind - 1, linenoise_ind + 1])
        difference = psd[:, linenoise_around] - \
            psd[:, linenoise_ind:linenoise_ind + 1]
        notch_ind = np.all(difference > 5, 1)

        if np.any(notch_ind):
            psd[notch_ind, linenoise_ind] = np.mean(
                psd[notch_ind, linenoise_around], axis=1)

    # Normalize
    psd = psd / np.max(np.abs(psd))

    psd = np.expand_dims(psd, (2, 3))
    psd = np.transpose(psd, [2, 1, 3, 0])

    # Autocorrelation
    autocorr = autocorr_fftw(icaact=icaact, trials=trials,
                             srate=srate, pnts=pnts, pct_data=pct_data)
    autocorr = np.expand_dims(autocorr, (2, 3))
    autocorr = np.transpose(autocorr, [2, 1, 3, 0])

    return [0.99 * topo, 0.99 * psd, 0.99 * autocorr]
