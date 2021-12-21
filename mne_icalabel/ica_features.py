import numpy as np
import math
import scipy.signal
import warnings

from .utils import gdatav4, pol2cart


def autocorr_fftw(icaact: np.array, sfreq: float) -> np.array:
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
    n_fft = 2**(math.ceil(math.log2(abs(2*n_times - 1))))
    ac = np.zeros((len(icaact), n_fft), dtype=np.float64)

    for idx in range(len(icaact)):
        X = np.fft.fft(icaact[idx:idx+1, :], n=n_fft, axis=1)
        ac[idx:idx+1, :] = np.mean(np.power(np.abs(X), 2), 2)

    # compute the inverse fourier transform as the auto-correlation
    ac = np.fft.ifft(ac, n=None, axis=1)  # ifft
    if n_times < sfreq:
        ac = np.hstack((ac[:, 0:n_times], np.zeros(
            (len(ac), sfreq - n_times + 1))))
    else:
        ac = ac[:, 0:sfreq+1]

    # normalize
    ac = ac[:, 0:sfreq+1] / ac[:, 0][:, None]

    # resample with polynomial interpolation
    resamp = scipy.signal.resample_poly(ac.T, 100, sfreq).T
    return resamp[:, 1:]


def rpsd(icaact: np.array,
             sfreq: float,
             pct_data: int = 100,
             subset=None) -> np.array:
    """Generates RPSD features for ICLabel.

    Parameters
    ----------
    icaact : np.array of shape (n_components, n_times, n_trials)
        The ICA activation waveforms produced from an ICA algorithm applied
        to data. The original data has shape of ``(n_channels, n_times)``,
        which is then decomposed into source space.
    sfreq : float
        The sampling rate of the Raw dataset in Hz.
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
    n_freqs = nyquist

    n_components, n_times, n_trials = icaact.shape
    n_points = min(n_times, sfreq)

    # define the Hamming window to perform convolution
    window = np.hamming(n_points).reshape(1, -1)[:, :, np.newaxis]
    cutoff = math.floor(n_times / n_points) * n_points
    index = np.ceil(np.arange(0, cutoff - n_points+1, n_points / 2)
                    ).astype(np.int64).reshape(1, -1) + \
        np.arange(0, n_points).reshape(-1, 1)

    # the number of segments
    n_seg = index.shape[1] * n_trials
    if subset is None:
        subset = np.random.permutation(
            n_seg)[:math.ceil(n_seg * pct_data / 100)]

    subset -= 1  # because matlab uses indices starting at 1
    subset = np.squeeze(subset)

    psd_feature = np.zeros((n_components, n_freqs))
    denom = sfreq * np.sum(np.power(window, 2))

    # compute the PSD for each ICA component
    for idx in range(n_components):
        temp = icaact[idx, index, :].reshape(
            1, index.shape[0], n_seg, order='F')
        temp = temp[:, :, subset] * window
        temp = scipy.signal.fft(temp, n_points, 1)
        temp = temp * np.conjugate(temp)
        temp = temp[:, 1:n_freqs + 1, :] * 2 / denom
        if n_freqs == nyquist:
            temp[:, -1, :] /= 2
        psd_feature[idx, :] = 20 * np.real(np.log10(np.median(temp, axis=2)))

    return psd_feature


def topoplot(icawinv: np.array, theta_coords: np.array,
                 rho_coords: np.array, picks: np.array=None) -> np.array_equal:
    """Generates topoplot image for ICLabel

    Parameters
    ----------
    icawinv : np.array of shape (n_components, n_components)
        pinv(EEG.icaweights*EEG.icasphere). The pseudoinverse of
        the ICA waveforms dot product with the ICA sphere values.
    theta_coords : np.array
        Theta coordinates of electrodes (polar)
    rho_coords : np.array
        Rho coordinates of electrodes (polar)
    picks : np.array
        Which channels to plot.

    Returns
    -------
    Z_i : np.array_equal
        Heatmap values as a (32 x 32 image).
    """
    GRID_SCALE = 32
    RMAX = 0.5
    theta_coords = np.atleast_2d(theta_coords)
    rho_coords = np.atleast_2d(rho_coords)
    print(theta_coords.shape)

    theta_coords = theta_coords * np.pi / 180
    allchansind = np.array(list(range(theta_coords.shape[1])))
    intchans = np.array(list(range(30)))
    if picks is None:
        picks = allchansind
    picks = np.squeeze(picks)
    x, y = pol2cart(theta_coords, rho_coords)
    allchansind = allchansind[picks]

    print(x.shape, y.shape)
    print(allchansind)
    rho_coords = rho_coords[:, picks]
    x = x[:, picks]
    y = y[:, picks]

    intx = x[:, intchans]
    inty = y[:, intchans]
    icawinv = icawinv[picks]
    intValues = icawinv[intchans]

    plotrad = min(1.0, np.max(rho_coords)*1.02)

    # Squeeze channel locations to <= RMAX
    squeezefac = RMAX / plotrad
    inty *= squeezefac
    intx *= squeezefac

    xi = np.linspace(-0.5, 0.5, GRID_SCALE)
    yi = np.linspace(-0.5, 0.5, GRID_SCALE)

    XQ, YQ = np.meshgrid(xi, yi)

    # Do interpolation with v4 scheme from MATLAB
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xi, Yi, Zi = gdatav4(inty, intx, intValues, YQ, XQ)

    mask = np.sqrt(np.power(Xi, 2) + np.power(Yi, 2)) > RMAX

    Zi[mask] = np.nan

    return Zi.T
