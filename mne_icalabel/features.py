import math
from typing import Union

from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA
import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import resample_poly


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
    topo = eeg_topoplot()

    # compute psd feature (float32)
    psd = eeg_rpsd()

    # compute autocorr feature (float32)
    if isinstance(inst, BaseRaw):
        if 5 < inst.times.size / inst.info['sfreq']:
            autocorr = eeg_autocorr_welch(inst, ica, icaact)
        else:
            autocorr = eeg_autocorr(inst, ica, icaact)
    else:
        autocorr = eeg_autocorr_fftw(inst, ica, icaact)


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
def next_power_of_2(x):
    """Equivalent to 2^nextpow2 in MATLAB."""
    return 1 if x == 0 else 2**(x - 1).bit_length()


def eeg_autocorr_welch(raw: BaseRaw, ica: ICA, icaact: np.ndarray):
    """Autocorrelation feature applied on raw object with at least 5 * fs
    samples (5 seconds).
    MATLAB: 'eeg_autocorr_welch.m'."""
    assert isinstance(raw, BaseRaw)  # sanity-check

    # in MATLAB, 'pct_data' variable is never provided and is always set to 100
    # and since it's only used in a code segment that is never executed..

    # setup constants
    ncomp = ica.n_components_
    n_points = min(raw.times.size, int(raw.info['sfreq'] * 3))
    nfft = next_power_of_2(2 * n_points - 1)
    cutoff = math.floor(raw.times.size / n_points) * n_points
    range_ = np.arange(0, cutoff - n_points + n_points / 2, n_points / 2)
    index = np.tile(range_, (n_points, 1)).T + np.arange(0, n_points)
    # python uses 0-index and matlab uses 1-index
    # thus (1:n_points) becomes np.arange(0, n_points)
    index = index.T.astype(int)

    # separate data segments
    """
    ## I misread.. and spent time on the part if pct_data != 100.. which never
    ## occurs. Just in case, here it is:

    n_seg = index.shape[1]
    # In MATLAB: n_seg = size(index, 2) * EEG.trials;
    # But EEG.trials is always provided as 1 because only raw dataset reach
    # this point.

    # In MATLAB: subset = randperm(n_seg, ceil(n_seg * pct_data / 100)); % need to find a better way to take subset
    # You don't say! 'pct_data' is always equal to 100..
    # In Python: I did not find a way to match randperm; the equivalent
    # function does not exist in numpy.
    # - np.random.permutation(range(1, n_seg + 1))
    # - np.random.choice(range(1, n_seg+1), size=5, replace=False)
    # - L = list(range(1, n_seg + 1))
    #   np.random.shuffle(L)
    # for a common np.random.seed().
    subset = np.random.permutation(range(n_seg))  # 0-index
    temp = np.hstack([icaact[:, index[:, k]] for k in range(index.shape[-1])])
    temp = temp.reshape(ncomp, *index.shape, order='F')
    segments = temp[:, :, subset]
    # I think that was basically to create segments with overlap.. but don't
    # quote me on that.
    """

    temp = np.hstack([icaact[:, index[:, k]] for k in range(index.shape[-1])])
    segments = temp.reshape(ncomp, *index.shape, order='F')

    # calc autocorrelation
    ac = np.zeros((ncomp, nfft))
    for it in range(ncomp):
        x = np.fft.fft(segments[it, :, :], nfft, axis=0)
        ac[it, :] = np.mean(np.power(np.abs(x), 2), axis=1)
    ac = np.fft.ifft(ac)

    # normalize
    # In MATLAB, there are 2 scenarios:
    # - EEG.pnts < EEG.srate: never occurs since we are using raw that last at
    # least 5 seconds
    # - EEG.pnts > EEG.srate
    ac = ac[:, :int(raw.info['sfreq']) + 1]
    # build that 3-line denominator
    arr1 = np.arange(n_points, n_points - int(raw.info['sfreq']), -1)
    arr1 = np.hstack([arr1, [np.max([1, n_points - int(raw.info['sfreq'])])]])
    den = np.tile(ac[:, 0], (arr1.size, 1))
    den = den.T * arr1 / n_points
    # finally..
    ac = np.divide(ac, den)

    # resample to 1 second at 100 samples/sec
    resamp = resample_poly(ac.T, 100, raw.info['sfreq']).T
    resamp = resamp[:, 1:, np.newaxis, np.newaxis].transpose([2, 1, 3, 0])
    return resamp.astype(np.float32)


def eeg_autocorr(raw: BaseRaw, ica: ICA, icaact: np.ndarray):
    """Autocorrelation feature applied on raw object that do not have enough
    sampes for eeg_autocorr_welch.
    MATLAB: 'eeg_autocorr.m'."""
    assert isinstance(raw, BaseRaw)  # sanity-check

    # in MATLAB, 'pct_data' variable is not used.
    ncomp = ica.n_components_
    nfft = next_power_of_2(2 * raw.times.size - 1)

    c = np.zeros((ncomp, nfft))
    for it in range(ncomp):
        # in MATLAB, 'mean' does nothing here. It looks like it was included
        # for a case where epochs are provided.
        x = np.power(np.abs(np.fft.fft(icaact[it, :], n=nfft)), 2)
        c[it, :] = np.fft.ifft(x)

    if raw.times.size < raw.info['sfreq']:
        zeros = np.zeros(
            (c.shape[0], int(raw.info['sfreq']) - raw.times.size + 1))
        ac = np.hstack([c[:, :raw.times.size], zeros])
    else:
        ac = c[:, :int(raw.info['sfreq']) + 1]

    # normalize by 0-tap autocorrelation
    ac = np.divide(ac.T, ac[:, 0]).T

    # resample to 1 second at 100 samples/sec
    resamp = resample_poly(ac.T, 100, raw.info['sfreq']).T
    resamp = resamp[:, 1:, np.newaxis, np.newaxis].transpose([2, 1, 3, 0])
    return resamp.astype(np.float32)


def eeg_autocorr_fftw(epochs: BaseEpochs, ica: ICA, icaact: np.ndarray):
    """Autocorrelation feature applied on epoch object.
    MATLAB: 'eeg_autocorr_fftw.m'."""
    assert isinstance(epochs, BaseEpochs)  # sanity-check

    # in MATLAB, 'pct_data' variable is not used.
    ncomp = ica.n_components_
    nfft = next_power_of_2(2 * epochs.times.size - 1)

    ac = np.zeros((ncomp, nfft))
    for it in range(ncomp):
        x = np.fft.fft(icaact[it, :, :], nfft, axis=0)
        ac[it, :] = np.mean(np.power(np.abs(x), 2), axis=1)

    ac = np.fft.ifft(ac)

    if epochs.times.size < epochs.info['sfreq']:
        zeros = np.zeros(
            (ac.shape[0], int(epochs.info['sfreq']) - epochs.times.size + 1))
        ac = np.hstack([ac[:, :epochs.times.size], zeros])
    else:
        ac = ac[:, :int(epochs.info['sfreq']) + 1]

    ac = np.divide(ac.T, ac[:, 0]).T

    # resample to 1 second at 100 samples/sec
    resamp = resample_poly(ac.T, 100, epochs.info['sfreq']).T
    resamp = resamp[:, 1:, np.newaxis, np.newaxis].transpose([2, 1, 3, 0])
    return resamp.astype(np.float32)
