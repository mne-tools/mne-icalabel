import numpy as np
import math
import scipy.signal
import warnings
from numpy.typing import ArrayLike
import mne

from .utils import gdatav4, pol2cart


def autocorr_fftw(icaact: ArrayLike, sfreq: float) -> ArrayLike:
    """
    Generates autocorrelation features for ICLabel.

    Parameters
    ----------
    icaact : np.array of shape (n_components, n_times, n_trials)
        The ICA activation waveforms produced from an ICA algorithm applied
        to data. The original data has shape of
        ``(n_channels, n_times)``,
        which is then decomposed into source space.
    sfreq : float
        The sampling rate of the Raw dataset in Hz.
    pct_data : int, optional
        [description], by default 100

    Returns
    -------
    resamp : np.array of shape (n_components, n_fft)
        The autocorrelation feature.
    """
    _, n_times, _ = icaact.shape
    sfreq = int(sfreq)

    # the number of FFTs and shape of the autocorrelation
    n_fft = 2 ** (math.ceil(math.log2(abs(2 * n_times - 1))))
    ac = np.zeros((len(icaact), n_fft), dtype=np.float64)

    for idx in range(len(icaact)):
        X = np.fft.fft(icaact[idx: idx + 1, :], n=n_fft, axis=1)
        ac[idx: idx + 1, :] = np.mean(np.power(np.abs(X), 2), 2)

    # compute the inverse fourier transform as the auto-correlation
    ac = np.fft.ifft(ac, n=None, axis=1)  # ifft
    if n_times < sfreq:
        ac = np.hstack((ac[:, 0:n_times], np.zeros(
            (len(ac), sfreq - n_times + 1))))
    else:
        ac = ac[:, 0: sfreq + 1]

    # normalize
    ac = ac[:, 0: sfreq + 1] / ac[:, 0][:, None]

    # resample with polynomial interpolation
    resamp = scipy.signal.resample_poly(ac.T, 100, sfreq).T
    return resamp[:, 1:]


def rpsd(icaact: ArrayLike, sfreq: int,
         pct_data: int = 100, subset=None) -> ArrayLike:
    """Generates RPSD features for ICLabel.

    Parameters
    ----------
    icaact : np.array of shape (n_components, n_times, n_trials)
        The ICA activation waveforms produced from an ICA algorithm applied
        to data. The original data has shape of ``(n_channels, n_times)``,
        which is then decomposed into source space.
    sfreq : int
        The sampling rate of the Raw dataset in Hz. Must be
        an integer.
    pct_data : int, optional
        [description], by default 100
    subset : [type], optional
        [description], by default None

    Returns
    -------
    psd_feature : np.array of shape (n_components, n_freqs)
        The power spectral density (PSD) feature of the ICA
        components.
    """

    # Clean input cutoff freq
    nyquist = math.floor(sfreq / 2)
    n_freqs = 100
    if n_freqs > nyquist:
        n_freqs = nyquist

    n_components, n_times, n_trials = icaact.shape
    n_points = min(n_times, sfreq)

    # define the Hamming window to perform convolution
    window = np.hamming(n_points).reshape(1, -1)[:, :, np.newaxis]
    cutoff = math.floor(n_times / n_points) * n_points
    index = np.ceil(np.arange(0, cutoff - n_points + 1, n_points / 2)).astype(
        np.int64
    ).reshape(1, -1) + np.arange(0, n_points).reshape(-1, 1)
    index = index.astype(np.int64)

    # the number of segments
    n_seg = index.shape[1] * n_trials
    if subset is None:
        subset = np.random.permutation(
            n_seg)[: math.ceil(n_seg * pct_data / 100)]

    # subset -= 1  # because matlab uses indices starting at 1
    subset = np.squeeze(subset)
    psd_feature = np.zeros((n_components, n_freqs))
    denom = sfreq * np.sum(np.power(window, 2))

    # compute the PSD for each ICA component
    for idx in range(n_components):
        temp = icaact[idx, index, :]
        temp = temp.reshape(1, index.shape[0], n_seg, order="F")
        temp = temp[:, :, subset] * window
        temp = scipy.fft.fft(temp, n_points, 1)
        temp = temp * np.conjugate(temp)
        temp = temp[:, 1: n_freqs + 1, :] * 2 / denom
        if n_freqs == nyquist:
            temp[:, -1, :] /= 2

        # convert PSD to decibals (20 * log(power))
        psd_feature[idx, :] = 20 * np.real(np.log10(np.median(temp, axis=2)))

    nfreq = psd_feature.shape[1]
    if nfreq < 100:
        psd_feature = np.concatenate(
            [psd_feature, np.tile(psd_feature[:, -1:], (1, 100 - nfreq))],
            axis=1
        )

    for linenoise_ind in [50, 60]:
        linenoise_around = np.array([linenoise_ind - 1, linenoise_ind + 1])
        difference = (
            psd_feature[:, linenoise_around] -
            psd_feature[:, linenoise_ind: linenoise_ind + 1]
        )
        notch_ind = np.all(difference > 5, 1)

        if np.any(notch_ind):
            psd_feature[notch_ind, linenoise_ind] = np.mean(
                psd_feature[notch_ind, linenoise_around], axis=1
            )

    # # Normalize
    psd_feature = psd_feature / \
        np.max(np.abs(psd_feature), axis=1, keepdims=True)

    psd_feature = np.expand_dims(psd_feature, (2, 3))
    psd_feature = np.transpose(psd_feature, [2, 1, 3, 0])

    return psd_feature


def topoplot(
    icawinv: ArrayLike,
    theta_coords: ArrayLike,
    rho_coords: ArrayLike,
    picks: ArrayLike = None,
) -> ArrayLike:
    """Generates topoplot image for ICLabel

    Parameters
    ----------
    icawinv : np.array of shape (n_components, n_components)
        pinv(EEG.icaweights*EEG.icasphere). The pseudoinverse of
        the ICA waveforms dot product with the ICA sphere values.
    theta_coords : np.array of shape (1, n_channels)
        Theta coordinates of electrodes (polar)
    rho_coords : np.array of shape (1, n_channels)
        Rho coordinates of electrodes (polar)
    picks : np.array of shape (n_channels)
        Which channels to plot.

    Returns
    -------
    Z_i : np.array
        Heatmap values as a (32 x 32 image).
    """
    GRID_SCALE = 32
    RMAX = 0.5

    if picks is None:
        n_chs = len(rho_coords.flatten())
        picks = np.arange(n_chs)

    rho_coords = rho_coords[:, picks]
    theta_coords = theta_coords[:, picks]

    # check if there are nans
    # if any(np.isnan(coords).any() for coords in [rho_coords, theta_coords]):
    #     # find nan inds
    #     rho_notnan_idx = np.argwhere(~np.isnan(rho_coords.flatten()))
    #     theta_notnan_idx = np.argwhere(~np.isnan(theta_coords.flatten()))

    #     assert_array_equal(rho_notnan_idx, theta_notnan_idx)
    #     # remove nans
    #     rho_coords = rho_coords[:, rho_notnan_idx]
    #     theta_coords = theta_coords[:, theta_notnan_idx]

    # convert to radians
    theta_coords = theta_coords * np.pi / 180
    intchans = np.arange(0, len(picks))

    # get cartesian coordinates
    x, y = pol2cart(theta_coords, rho_coords)

    intx = x[:, intchans]
    inty = y[:, intchans]
    icawinv = icawinv[picks]
    intValues = icawinv[intchans]

    plotrad = min(1.0, np.max(rho_coords) * 1.02)

    # Squeeze channel locations to <= RMAX
    squeezefac = RMAX / plotrad
    inty *= squeezefac
    intx *= squeezefac

    xi = np.linspace(-0.5, 0.5, GRID_SCALE)
    yi = np.linspace(-0.5, 0.5, GRID_SCALE)

    XQ, YQ = np.meshgrid(xi, yi)

    # Do interpolation with v4 scheme from MATLAB
    # see: https://github.com/yjmantilla/gdatav4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xi, Yi, Zi = gdatav4(inty, intx, intValues, YQ, XQ)

    # mask outside the head
    mask = np.sqrt(np.power(Xi, 2) + np.power(Yi, 2)) > RMAX
    Zi[mask] = np.nan

    return Zi.T


def mne_to_eeglab_locs(raw: mne.io.BaseRaw):
    """Obtain EEGLab-like spherical coordinate from EEG channel positions.

    TODO: @JACOB:
    - Where is (0,0,0) defined in MNE vs EEGLab?
    - some text description of how the sphere coordinates differ between MNE
    and EEGLab.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Instance of raw object with a `mne.montage.DigMontage` set with
        ``n_channels`` channel positions.

    Returns
    -------
    Rd : np.array of shape (1, n_channels)
        Angle in spherical coordinates of each EEG channel.
    Th : np.array of shape (1, n_channels)
        Degree in spherical coordinates of each EEG channel.
    """

    def _sph2topo(_theta, _phi):
        """
        Convert spherical coordinates to topo.
        """
        az = _phi
        horiz = _theta
        angle = -1 * horiz
        radius = (np.pi / 2 - az) / np.pi
        return angle, radius

    def _cart2sph(_x, _y, _z):
        """
        Convert cartesian coordinates to spherical.
        """
        azimuth = np.arctan2(_y, _x)
        elevation = np.arctan2(_z, np.sqrt(_x**2 + _y**2))
        r = np.sqrt(_x**2 + _y**2 + _z**2)
        # theta,phi,r
        return azimuth, elevation, r

    # get the channel position dictionary
    montage = raw.get_montage()
    positions = montage.get_positions()
    ch_pos = positions["ch_pos"]

    # get locations as a 2D array
    locs = np.vstack(list(ch_pos.values()))

    # Obtain carthesian coordinates
    x = locs[:, 1]

    # be mindful of the nose orientation in eeglab and mne
    # TODO: @Jacob, please expand on this.
    y = -1 * locs[:, 0]
    # see https://github.com/mne-tools/mne-python/blob/24377ad3200b6099ed47576e9cf8b27578d571ef/mne/io/eeglab/eeglab.py#L105  # noqa
    z = locs[:, 2]

    # Obtain Spherical Coordinates
    sph = np.array([_cart2sph(x[i], y[i], z[i]) for i in range(len(x))])
    theta = sph[:, 0]
    phi = sph[:, 1]

    # Obtain Polar coordinates (as in eeglab)
    topo = np.array([_sph2topo(theta[i], phi[i]) for i in range(len(theta))])
    rd = topo[:, 1]
    th = topo[:, 0]

    return rd.reshape([1, -1]), np.degrees(th).reshape([1, -1])
