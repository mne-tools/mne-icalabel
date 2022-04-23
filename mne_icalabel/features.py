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
        if 5 < inst.times.size / inst.info["sfreq"]:
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
def eeg_rpsd(inst: Union[BaseRaw, BaseEpochs], ica: ICA, icaact: np.ndarray):
    """PSD feature."""
    assert isinstance(inst, (BaseRaw, BaseEpochs))  # sanity-check

    constants = _eeg_rpsd_constants(inst, ica)
    psd = _eeg_rpsd_compute_psdmed(inst, icaact, *constants)

    # extrapolate or prune as needed
    nfreq = psd.shape[1]
    if nfreq < 100:
        psd = np.concatenate([psd, np.tile(psd[:, -1:], (1, 100 - nfreq))], axis=1)

    # undo notch filter


    return psd


def _eeg_rpsd_constants(inst: Union[BaseRaw, BaseEpochs], ica: ICA):
    """Computes the constants before ``randperm`` is used to compute the
    subset."""
    # in MATLAB, 'pct_data' variable is never provided and is always initialized
    # to 100. 'pct_data' is only used in a division by 100.. and thus has no
    # impact and is omitted here.
    # in MATLAB, 'nfreqs' variable is always provided as 100 to this function,
    # thus it is either equal to 100 or to the nyquist frequency depending on
    # the nyquist frequency.

    nyquist = np.floor(inst.info['sfreq'] / 2).astype(int)
    nfreqs = nyquist if nyquist < 100 else 100

    ncomp = ica.n_components_
    n_points = min(inst.times.size, int(inst.info["sfreq"]))
    window = np.hamming(n_points)
    cutoff = np.floor(inst.times.size / n_points) * n_points

    # python is 0-index while matlab is 1-index, thus (1:n_points) becomes
    # np.arange(0, n_points) since 'index' is used to select from arrays.
    range_ = np.ceil(np.arange(0, cutoff - n_points + n_points / 2, n_points / 2))
    index = np.tile(range_, (n_points, 1)).T + np.arange(0, n_points)
    index = index.T.astype(int)

    # different behaviors based on EEG.trials, i.e. raw or epoch
    if isinstance(inst, BaseRaw):
        n_seg = index.shape[1]
    if isinstance(inst, BaseEpochs):
        n_seg = index.shape[1] * len(inst)

    # in MATLAB: 'subset = randperm(n_seg, ceil(n_seg * pct_data / 100));'
    # which is basically: 'subset = randperm(n_seg, n_seg);'
    # np.random.seed() can be used to fix the seed to the same value as MATLAB,
    # but since the 'randperm' equivalent in numpy does not exist, it is not
    # possible to reproduce the output in python.
    # 'subset' is used to select from arrays and is 0-index in Python while its
    # 1-index in MATLAB.
    subset = np.random.permutation(range(n_seg))  # 0-index

    return ncomp, nfreqs, n_points, nyquist, index, window, subset


def _eeg_rpsd_compute_psdmed(
        inst: Union[BaseRaw, BaseEpochs],
        icaact: np.ndarray,
        ncomp: int,
        nfreqs: int,
        n_points: int,
        nyquist: int,
        index: np.ndarray,
        window: np.ndarray,
        subset:np.ndarray,
        ) -> np.ndarray:
    """Compute the variable 'psdmed', annotated as windowed spectrums."""
    denominator = inst.info['sfreq'] * np.sum(np.power(window, 2))
    psdmed = np.zeros((ncomp, nfreqs))
    for it in range(ncomp):
        # Compared to MATLAB, shapes differ as the component dimension (size 1)
        # was squeezed.
        temp = np.hstack([icaact[it, index[:, k]] for k in range(index.shape[-1])])
        temp = temp.reshape(*index.shape, order="F")
        temp = (temp[:, subset].T * window).T
        temp = np.fft.fft(temp, n_points, axis=0)
        temp = temp * np.conjugate(temp)
        temp = temp[1:nfreqs + 1, :] * 2 / denominator
        if nfreqs == nyquist:
            temp[-1, :] = temp[-1, :] / 2
        psdmed[it, :] = 20 * np.real(np.log10(np.median(temp, axis=1)))

    return psdmed



# ----------------------------------------------------------------------------
def next_power_of_2(x):
    """Equivalent to 2^nextpow2 in MATLAB."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def eeg_autocorr_welch(raw: BaseRaw, ica: ICA, icaact: np.ndarray):
    """Autocorrelation feature applied on raw object with at least 5 * fs
    samples (5 seconds).
    MATLAB: 'eeg_autocorr_welch.m'."""
    assert isinstance(raw, BaseRaw)  # sanity-check

    # in MATLAB, 'pct_data' variable is never provided and is always initialized
    # to 100. 'pct_data' is only used in an 'if' statement reached if 'pct_data'
    # is different than 100.. thus, 'pct_data' is not used by this autocorrelation
    # function and is omitted here.

    # setup constants
    ncomp = ica.n_components_
    n_points = min(raw.times.size, int(raw.info["sfreq"] * 3))
    nfft = next_power_of_2(2 * n_points - 1)
    cutoff = np.floor(raw.times.size / n_points) * n_points
    range_ = np.ceil(np.arange(0, cutoff - n_points + n_points / 2, n_points / 2))
    index = np.tile(range_, (n_points, 1)).T + np.arange(0, n_points)
    # python uses 0-index and matlab uses 1-index
    # python is 0-index while matlab is 1-index, thus (1:n_points) becomes
    # np.arange(0, n_points) since 'index' is used to select from arrays.
    index = index.T.astype(int)

    # separate data segments
    temp = np.hstack([icaact[:, index[:, k]] for k in range(index.shape[-1])])
    segments = temp.reshape(ncomp, *index.shape, order="F")

    """
    # Just in case, here is the 'if' statement when 'pct_data' is different
    # than 100.

    n_seg = index.shape[1]
    # In MATLAB: n_seg = size(index, 2) * EEG.trials;
    # However, this function is only called on RAW dataset with EEG.trials
    # equal to 1.

    # in MATLAB: 'subset = randperm(n_seg, ceil(n_seg * pct_data / 100));'
    # which is basically: 'subset = randperm(n_seg, n_seg);'
    # np.random.seed() can be used to fix the seed to the same value as MATLAB,
    # but since the 'randperm' equivalent in numpy does not exist, it is not
    # possible to reproduce the output in python.
    # 'subset' is used to select from arrays and is 0-index in Python while its
    # 1-index in MATLAB.
    subset = np.random.permutation(range(n_seg))  # 0-index
    temp = np.hstack([icaact[:, index[:, k]] for k in range(index.shape[-1])])
    temp = temp.reshape(ncomp, *index.shape, order='F')
    segments = temp[:, :, subset]
    """

    # calc autocorrelation
    ac = np.zeros((ncomp, nfft))
    for it in range(ncomp):
        x = np.fft.fft(segments[it, :, :], nfft, axis=0)
        ac[it, :] = np.mean(np.power(np.abs(x), 2), axis=1)
    ac = np.fft.ifft(ac)

    # normalize
    # In MATLAB, 2 scenarios are defined:
    # - EEG.pnts < EEG.srate, which never occurs since then raw provided to
    # this autocorrelation function last at least 5 second.
    # - EEG.pnts > EEG.srate, implemented below.
    ac = ac[:, : int(raw.info["sfreq"]) + 1]
    # build the (3-line!) denominator
    arr1 = np.arange(n_points, n_points - int(raw.info["sfreq"]), -1)
    arr1 = np.hstack([arr1, [np.max([1, n_points - int(raw.info["sfreq"])])]])
    den = np.tile(ac[:, 0], (arr1.size, 1))
    den = den.T * arr1 / n_points
    # finally..
    ac = np.divide(ac, den)

    # resample to 1 second at 100 samples/sec
    resamp = resample_poly(ac.T, 100, raw.info["sfreq"]).T
    resamp = resamp[:, 1:, np.newaxis, np.newaxis].transpose([2, 1, 3, 0])
    return resamp.astype(np.float32)


def eeg_autocorr(raw: BaseRaw, ica: ICA, icaact: np.ndarray):
    """Autocorrelation feature applied on raw object that do not have enough
    sampes for eeg_autocorr_welch.
    MATLAB: 'eeg_autocorr.m'."""
    assert isinstance(raw, BaseRaw)  # sanity-check

    # in MATLAB, 'pct_data' variable is neither provided or used, thus it is
    # omitted here.
    ncomp = ica.n_components_
    nfft = next_power_of_2(2 * raw.times.size - 1)

    c = np.zeros((ncomp, nfft))
    for it in range(ncomp):
        # in MATLAB, 'mean' does nothing here. It looks like it was included
        # for a case where epochs are provided, which never happens with this
        # autocorrelation function.
        x = np.power(np.abs(np.fft.fft(icaact[it, :], n=nfft)), 2)
        c[it, :] = np.fft.ifft(x)

    if raw.times.size < raw.info["sfreq"]:
        zeros = np.zeros((c.shape[0], int(raw.info["sfreq"]) - raw.times.size + 1))
        ac = np.hstack([c[:, : raw.times.size], zeros])
    else:
        ac = c[:, : int(raw.info["sfreq"]) + 1]

    # normalize by 0-tap autocorrelation
    ac = np.divide(ac.T, ac[:, 0]).T

    # resample to 1 second at 100 samples/sec
    resamp = resample_poly(ac.T, 100, raw.info["sfreq"]).T
    resamp = resamp[:, 1:, np.newaxis, np.newaxis].transpose([2, 1, 3, 0])
    return resamp.astype(np.float32)


def eeg_autocorr_fftw(epochs: BaseEpochs, ica: ICA, icaact: np.ndarray):
    """Autocorrelation feature applied on epoch object.
    MATLAB: 'eeg_autocorr_fftw.m'."""
    assert isinstance(epochs, BaseEpochs)  # sanity-check

    # in MATLAB, 'pct_data' variable is neither provided or used, thus it is
    # omitted here.
    ncomp = ica.n_components_
    nfft = next_power_of_2(2 * epochs.times.size - 1)

    ac = np.zeros((ncomp, nfft))
    for it in range(ncomp):
        x = np.fft.fft(icaact[it, :, :], nfft, axis=0)
        ac[it, :] = np.mean(np.power(np.abs(x), 2), axis=1)

    ac = np.fft.ifft(ac)

    if epochs.times.size < epochs.info["sfreq"]:
        zeros = np.zeros(
            (ac.shape[0], int(epochs.info["sfreq"]) - epochs.times.size + 1)
        )
        ac = np.hstack([ac[:, : epochs.times.size], zeros])
    else:
        ac = ac[:, : int(epochs.info["sfreq"]) + 1]

    ac = np.divide(ac.T, ac[:, 0]).T

    # resample to 1 second at 100 samples/sec
    resamp = resample_poly(ac.T, 100, epochs.info["sfreq"]).T
    resamp = resamp[:, 1:, np.newaxis, np.newaxis].transpose([2, 1, 3, 0])
    return resamp.astype(np.float32)
