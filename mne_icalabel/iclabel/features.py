from typing import List, Tuple, Union

import numpy as np
from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA
from mne.utils import warn
from numpy.typing import NDArray
from scipy.signal import resample_poly

from ..utils._checks import _validate_inst_and_ica
from .utils import _gdatav4, _mne_to_eeglab_locs, _next_power_of_2, _pol2cart


def get_iclabel_features(inst: Union[BaseRaw, BaseEpochs], ica: ICA):
    """Generate the features for ICLabel neural network.

    Parameters
    ----------
    inst : Raw | Epochs
        MNE Raw/Epoch instance with data array in Volts.
    ica : ICA
        MNE ICA decomposition.

    Returns
    -------
    topo : array of shape (32, 32, 1, n_components)
        The topoplot feature.
    psd : array of shape (1, 100, 1, n_components)
        The psd feature.
    autocorr : array of shape (1, 100, 1, n_components)
        The autocorrelations feature. Depending on the length of the
        raw data passed in, different methods of computing autocorrelation
        will be used. See :footcite:`iclabel2019` for details.

    References
    ----------
    .. footbibliography::
    """
    _validate_inst_and_ica(inst, ica)

    # TODO: 'custom_ref_applied' does not necessarily correspond to a CAR reference.
    # At the moment, the reference of the EEG data is not stored in the info.
    # c.f. https://github.com/mne-tools/mne-python/issues/8962
    if inst.info["custom_ref_applied"] == 0:
        warn(
            f"The provided {'Raw' if isinstance(inst, BaseRaw) else 'Epochs'} instance "
            "does not seem to be referenced to a common average reference (CAR). "
            "ICLabel was designed to classify features extracted from an EEG dataset "
            "referenced to a CAR (see the 'set_eeg_reference()' method for Raw and "
            "Epochs instances)."
        )
    if inst.info["highpass"] != 1 or inst.info["lowpass"] != 100:
        warn(
            f"The provided {'Raw' if isinstance(inst, BaseRaw) else 'Epochs'} instance "
            "is not filtered between 1 and 100 Hz. "
            "ICLabel was designed to classify features extracted from an EEG dataset "
            "bandpass filtered between 1 and 100 Hz (see the 'filter()' method for Raw "
            "and Epochs instances)."
        )
    # confirm that the ICA uses an infomax extended
    method_ = ica.method not in ("infomax", "picard")
    extended_ = "extended" not in ica.fit_params or ica.fit_params["extended"] is False
    ortho_ = "ortho" not in ica.fit_params or ica.fit_params["ortho"] is True
    ortho_ = ortho_ if ica.method == "picard" else False
    if any((method_, extended_, ortho_)):
        warn(
            f"The provided ICA instance was fitted with a '{ica.method}' algorithm. "
            "ICLabel was designed with extended infomax ICA decompositions. To use the "
            "extended infomax algorithm, use the 'mne.preprocessing.ICA' instance with the "
            "arguments 'ICA(method='infomax', fit_params=dict(extended=True))' (scikit-learn) or "
            "'ICA(method='picard', fit_params=dict(ortho=False, extended=True))' (python-picard)."
        )

    icawinv, _ = _retrieve_eeglab_icawinv(ica)
    icaact = _compute_ica_activations(inst, ica)

    # compute topographic feature (float32)
    topo = _eeg_topoplot(inst, icawinv, ica.ch_names)

    # compute psd feature (float32)
    psd = _eeg_rpsd(inst, ica, icaact)

    # compute autocorr feature (float32)
    if isinstance(inst, BaseRaw):
        if 5 < inst.times.size / inst.info["sfreq"]:
            autocorr = _eeg_autocorr_welch(inst, ica, icaact)
        else:
            autocorr = _eeg_autocorr(inst, ica, icaact)
    else:
        autocorr = _eeg_autocorr_fftw(inst, ica, icaact)

    # scale by 0.99
    topo *= 0.99
    psd *= 0.99
    autocorr *= 0.99

    return topo, psd, autocorr


def _retrieve_eeglab_icawinv(
    ica: ICA,
) -> Tuple[NDArray[float], NDArray[float]]:
    """
    Retrieve 'icawinv' from an MNE ICA instance.

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
    icawinv = np.linalg.pinv(weights)
    # sanity-check
    assert icawinv.shape[-1] == ica.n_components_
    assert icawinv.ndim == 2
    return icawinv, weights


def _compute_ica_activations(inst: Union[BaseRaw, BaseEpochs], ica: ICA) -> NDArray[float]:
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

    Notes
    -----
    In EEGLAB, the ICA activations are computed after the data and the ICA
    decomposition are re-referenced to a common average, if the field 'EEG.ref'
    is different from 'averef'. The EEGLAB sample dataset's field 'EEG.ref' is
    set to 'common', thus triggering the re-referencing with 'pop_reref' which
    seems to be buggy and breaks the ICA solution. After 'pop_reref' is called,
    the relation 'inv(EEG.icaweights * EEG.icasphere) = EEG.icawinv' is not
    respected anymore.

    Additionally, 'pop_reref' changes the field 'EEG.ref' to 'average'. It is
    assumed that 'common', 'average' and 'averef' are all denoting a common
    average reference.
    """
    icawinv, weights = _retrieve_eeglab_icawinv(ica)
    icasphere = np.eye(icawinv.shape[0])
    data = inst.get_data(picks=ica.ch_names) * 1e6
    icaact = (weights[0 : ica.n_components_, :] @ icasphere) @ data
    # move trial (epoch) dimension to the end
    if icaact.ndim == 3:
        assert isinstance(inst, BaseEpochs)  # sanity-check
        icaact = icaact.transpose([1, 2, 0])
    return icaact


# ----------------------------------------------------------------------------
def _eeg_topoplot(
    inst: Union[BaseRaw, BaseEpochs], icawinv: NDArray[float], picks: List[str]
) -> NDArray[float]:
    """Topoplot feature."""
    ncomp = icawinv.shape[-1]
    topo = np.zeros((32, 32, 1, ncomp))
    rd, th = _mne_to_eeglab_locs(inst, picks)
    th = np.pi / 180 * th  # convert degrees to radians
    for it in range(ncomp):
        temp_topo = _topoplotFast(icawinv[:, it], rd, th)
        np.nan_to_num(temp_topo, copy=False)  # set NaN values to 0 in-place
        topo[:, :, 0, it] = temp_topo / np.max(np.abs(temp_topo))
    return topo.astype(np.float32)


def _topoplotFast(values: NDArray[float], rd: NDArray[float], th: NDArray[float]) -> NDArray[float]:
    """Implement topoplotFast.m from MATLAB. Each topographic map is a 32x32 images."""
    # constants
    GRID_SCALE = 32  # number of pixels
    rmax = 0.5  # actual head radius

    # convert electrode locations from polar to cartesian coordinates
    x, y = _pol2cart(th, rd)

    # prepare coordinates
    # Comments in MATLAB (L750:753) are:
    #   default: just outside the outermost electrode location
    #   default: plot out to the 0.5 head boundary
    #   don't plot channels with Rd > 1 (below head)
    plotrad = min(1, np.max(rd) * 1.02)
    plotrad = max(plotrad, 0.5)

    # TODO: Selection of channels.
    # For interpolation, only the channels inside the interpolation square are
    # considered. c.f. L839:843.

    # Squeeze channel location to <= rmax
    # Comments in MATLAB (L894:908)
    #   squeeze electrode arc_lengths towards the vertex to plot all inside the
    #   head cartoon
    squeezefac = rmax / plotrad
    rd *= squeezefac
    x *= squeezefac
    y *= squeezefac
    # convert to float63
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    # Find limits for interpolation
    xmin = min(-rmax, np.min(x))
    xmax = max(rmax, np.max(x))
    ymin = min(-rmax, np.min(y))
    ymax = max(rmax, np.max(y))

    # Interpolate scalp map data
    xi = np.linspace(xmin, xmax, GRID_SCALE).astype(np.float64).reshape((1, -1))
    yi = np.linspace(ymin, ymax, GRID_SCALE).astype(np.float64).reshape((1, -1))
    # additional step for gdatav4 compared to MATLAB: linspace to meshgrid
    XQ, YQ = np.meshgrid(xi, yi)
    Xi, Yi, Zi = _gdatav4(x, y, values.reshape((-1, 1)), XQ, YQ)
    # additional step for gdatav4 compared to MATLAB: transpose
    Zi = Zi.T

    # Mask out data outside the head
    mask = np.sqrt(np.power(Xi, 2) + np.power(Yi, 2)) <= rmax
    Zi[~mask] = np.nan

    return Zi


# ----------------------------------------------------------------------------
def _eeg_rpsd(inst: Union[BaseRaw, BaseEpochs], ica: ICA, icaact: NDArray[float]) -> NDArray[float]:
    """PSD feature."""
    assert isinstance(inst, (BaseRaw, BaseEpochs))  # sanity-check
    constants = _eeg_rpsd_constants(inst, ica)
    psd = _eeg_rpsd_compute_psdmed(inst, icaact, *constants)
    psd = _eeg_rpsd_format(psd)
    return psd


def _eeg_rpsd_constants(
    inst: Union[BaseRaw, BaseEpochs],
    ica: ICA,
) -> Tuple[int, int, int, int, NDArray[int], NDArray[float], NDArray[int]]:
    """Compute the constants before ``randperm`` is used to compute the subset."""
    # in MATLAB, 'pct_data' variable is never provided and is always initialized
    # to 100. 'pct_data' is only used in a division by 100.. and thus has no
    # impact and is omitted here.
    # in MATLAB, 'nfreqs' variable is always provided as 100 to this function,
    # thus it is either equal to 100 or to the nyquist frequency depending on
    # the nyquist frequency.

    nyquist = np.floor(inst.info["sfreq"] / 2).astype(int)
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
    icaact: NDArray[float],
    ncomp: int,
    nfreqs: int,
    n_points: int,
    nyquist: int,
    index: NDArray[int],
    window: NDArray[float],
    subset: NDArray[int],
) -> NDArray[float]:
    """Compute the variable 'psdmed', annotated as windowed spectrums."""
    denominator = inst.info["sfreq"] * np.sum(np.power(window, 2))
    psdmed = np.zeros((ncomp, nfreqs))
    for it in range(ncomp):
        # Compared to MATLAB, shapes differ as the component dimension (size 1)
        # was squeezed.
        if isinstance(inst, BaseRaw):
            temp = np.hstack([icaact[it, index[:, k]] for k in range(index.shape[-1])])
            temp = temp.reshape(*index.shape, order="F")
            # equivalent to:
            # np.vstack([icaact[it, index[:, k]] for k in range(index.shape[-1])]).T
        elif isinstance(inst, BaseEpochs):
            temp = np.vstack([icaact[it, index[:, k], :] for k in range(index.shape[-1])])
            temp = temp.reshape(index.shape[0], index.shape[1] * len(inst), order="F")
        temp = (temp[:, subset].T * window).T
        temp = np.fft.fft(temp, n_points, axis=0)
        temp = temp * np.conjugate(temp)
        temp = temp[1 : nfreqs + 1, :] * 2 / denominator
        if nfreqs == nyquist:
            temp[-1, :] = temp[-1, :] / 2
        psdmed[it, :] = 20 * np.real(np.log10(np.median(temp, axis=1)))

    return psdmed


def _eeg_rpsd_format(
    psd: NDArray[float],
) -> NDArray[float]:
    """Apply the formatting steps after 'eeg_rpsd.m' from the MATLAB feature extraction."""
    # extrapolate or prune as needed
    nfreq = psd.shape[1]
    if nfreq < 100:
        psd = np.concatenate([psd, np.tile(psd[:, -1:], (1, 100 - nfreq))], axis=1)

    # undo notch filter
    for linenoise_ind in (50, 60):
        # 'linenoise_ind' is used for array selection in psd, which is
        # 0-index in Python and 1-index in MATLAB.
        linenoise_ind -= 1
        linenoise_around = np.array([linenoise_ind - 1, linenoise_ind + 1])
        # 'linenoise_around' is used for array selection in psd, which is
        # 0-index in Python and 1-index in MATLAB.
        difference = (psd[:, linenoise_around].T - psd[:, linenoise_ind]).T
        notch_ind = np.all(5 < difference, axis=1)
        if any(notch_ind):
            # Numpy doesn't like the selection '[notch_ind, linenoise_ind]' with
            # 'notch_ind' as a bool mask. 'notch_ind' is first converted to int.
            # Numpy doesn't like the selection '[notch_ind, linenoise_around]'
            # with both defined as multi-values 1D arrays (or list). To get
            # around, the syntax [notch_ind[:, None], linenoise_around] is used.
            # That syntax works only with arrays (not list).
            notch_ind = np.where(notch_ind)[0]
            psd[notch_ind, linenoise_ind] = np.mean(
                psd[notch_ind[:, None], linenoise_around], axis=-1
            )

    # normalize
    psd = np.divide(psd.T, np.max(np.abs(psd), axis=-1)).T
    # reshape and cast
    return psd[:, :, np.newaxis, np.newaxis].transpose([2, 1, 3, 0]).astype(np.float32)


def _eeg_autocorr_welch(raw: BaseRaw, ica: ICA, icaact: NDArray[float]) -> NDArray[float]:
    """Autocorrelation feature applied on raw object with at least 5 * fs samples (5 seconds).

    MATLAB: 'eeg_autocorr_welch.m'.
    """
    assert isinstance(raw, BaseRaw)  # sanity-check

    # in MATLAB, 'pct_data' variable is never provided and is always initialized
    # to 100. 'pct_data' is only used in an 'if' statement reached if 'pct_data'
    # is different than 100.. thus, 'pct_data' is not used by this autocorrelation
    # function and is omitted here.

    # setup constants
    ncomp = ica.n_components_
    n_points = min(raw.times.size, int(raw.info["sfreq"] * 3))
    nfft = _next_power_of_2(2 * n_points - 1)
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
    resamp = _resample(ac, raw.info["sfreq"])
    resamp = resamp[:, 1:, np.newaxis, np.newaxis].transpose([2, 1, 3, 0])
    return np.real(resamp).astype(np.float32)


def _eeg_autocorr(raw: BaseRaw, ica: ICA, icaact: NDArray[float]) -> NDArray[float]:
    """Autocorrelation feature applied on raw object without enough sampes for eeg_autocorr_welch.

    MATLAB: 'eeg_autocorr.m'.
    """
    assert isinstance(raw, BaseRaw)  # sanity-check

    # in MATLAB, 'pct_data' variable is neither provided or used, thus it is
    # omitted here.
    ncomp = ica.n_components_
    nfft = _next_power_of_2(2 * raw.times.size - 1)

    c = np.zeros((ncomp, nfft))
    for it in range(ncomp):
        # in MATLAB, 'mean' does nothing here. It looks like it was included
        # for a case where epochs are provided, which never happens with this
        # autocorrelation function.
        x = np.power(np.abs(np.fft.fft(icaact[it, :], n=nfft)), 2)
        c[it, :] = np.real(np.fft.ifft(x))

    if raw.times.size < raw.info["sfreq"]:
        zeros = np.zeros((c.shape[0], int(raw.info["sfreq"]) - raw.times.size + 1))
        ac = np.hstack([c[:, : raw.times.size], zeros])
    else:
        ac = c[:, : int(raw.info["sfreq"]) + 1]

    # normalize by 0-tap autocorrelation
    ac = np.divide(ac.T, ac[:, 0]).T

    # resample to 1 second at 100 samples/sec
    resamp = _resample(ac, raw.info["sfreq"])
    resamp = resamp[:, 1:, np.newaxis, np.newaxis].transpose([2, 1, 3, 0])
    return resamp.astype(np.float32)


def _eeg_autocorr_fftw(epochs: BaseEpochs, ica: ICA, icaact: NDArray[float]) -> NDArray[float]:
    """Autocorrelation feature applied on epoch object.

    MATLAB: 'eeg_autocorr_fftw.m'.
    """
    assert isinstance(epochs, BaseEpochs)  # sanity-check

    # in MATLAB, 'pct_data' variable is neither provided or used, thus it is
    # omitted here.
    ncomp = ica.n_components_
    nfft = _next_power_of_2(2 * epochs.times.size - 1)

    ac = np.zeros((ncomp, nfft))
    for it in range(ncomp):
        x = np.fft.fft(icaact[it, :, :], nfft, axis=0)
        ac[it, :] = np.mean(np.power(np.abs(x), 2), axis=1)

    ac = np.fft.ifft(ac)

    if epochs.times.size < epochs.info["sfreq"]:
        zeros = np.zeros((ac.shape[0], int(epochs.info["sfreq"]) - epochs.times.size + 1))
        ac = np.hstack([ac[:, : epochs.times.size], zeros])
    else:
        ac = ac[:, : int(epochs.info["sfreq"]) + 1]

    ac = np.divide(ac.T, ac[:, 0]).T

    # resample to 1 second at 100 samples/sec
    resamp = _resample(ac, epochs.info["sfreq"])
    resamp = resamp[:, 1:, np.newaxis, np.newaxis].transpose([2, 1, 3, 0])
    return np.real(resamp).astype(np.float32)


def _resample(ac: NDArray[float], fs: Union[int, float]) -> NDArray[float]:
    """Resample the autocorrelation feature.

    The comment in EEGLAB is:
        resample to 1 second at 100 samples/sec

    Which translates by: the output array must be of shape (n_comp, 101), thus
    the resampling up variable is set to 100, and down variable must respect:
        100 < ac.T.shape[0] * 100 / down <= 101
    If the instance sampling frequency is an integer, then down is equal to the
    sampling frequency.

    Parameters
    ----------
    ac : array
        Array of shape (n_comp, samples).
    fs : int | float
        Sampling frequency of the MNE instance.
    """
    down = int(fs)
    if 101 < ac.shape[1] * 100 / down:
        down += 1
    return resample_poly(ac.T, 100, down).T
