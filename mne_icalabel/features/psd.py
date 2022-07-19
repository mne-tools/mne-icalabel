import numpy as np
from mne.io.pick import _picks_to_idx
from mne.time_frequency.psd import psd_multitaper
from mne.utils import _check_option
from mne.viz.ica import _prepare_data_ica_properties


def get_psd(
    ica,
    inst,
    picks=None,
    reject="auto",
    reject_by_annotation=False,
    fmin=0,
    fmax=np.inf,
    tmin=None,
    tmax=None,
    bandwidth=None,
    adaptive=False,
    low_bias=True,
    proj=False,
    n_jobs=1,
    normalization="length",
    output="power",
    dB=True,
):
    """Compute the power spectral density of the ICA components.

    Parameters
    ----------
    ica : instance of mne.preprocessing.ICA
        The ICA solution.
    inst : instance of Epochs or Raw
    picks : str | list | slice | None
        In lists, channel type strings (e.g., ['meg', 'eeg']) will pick channels
        of those types, channel name strings (e.g., ['MEG0111', 'MEG2623'] will
        pick the given channels. Can also be the string values “all” to pick all
        channels, or “data” to pick data channels. None (default) will pick all
        channels. Note that channels in info['bads'] will be included if their
        names or indices are explicitly provided.
    reject :'auto' | dict | None
        Allows to specify rejection parameters used to drop epochs
        (or segments if continuous signal is passed as inst).
        If None, no rejection is applied. The default is 'auto',
        which applies the rejection parameters used when fitting
        the ICA object.
    %(reject_by_annotation_raw)s
    fmin : float
        Min frequency of interest.
    fmax : float
        Max frequency of interest.
    tmin : float | None
        Min time of interest.
    tmax : float | None
        Max time of interest.
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz. The default
        value is a window half-bandwidth of 4.
    adaptive :  bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).
    low_bias : bool
        Only use tapers with more than 90%% spectral concentration within
        bandwidth.
    proj : bool
        Apply SSP projection vectors. If inst is ndarray this is not used.
    %(n_jobs)s
    %(normalization)s
    output : str
        The format of the returned ``psds`` array. Can be either ``'complex'``
        or ``'power'``. If ``'power'``, the power spectral density is returned.
        If ``output='complex'``, the complex fourier coefficients are returned
        per taper.
    dB : bool
        Whether to plot spectrum in dB. Defaults to True.

    Returns
    -------
    psds_mean : ndarray, shape(n_channels, n_freqs)
         Mean of power spectral densities on channel.
    """
    picks = _picks_to_idx(ica.info, picks, "all")[: ica.n_components_]
    kind, dropped_indices, epochs_src, data = _prepare_data_ica_properties(
        inst, ica, reject_by_annotation=True, reject="auto"
    )
    Nyquist = inst.info["sfreq"] / 2.0
    fmax = min(inst.info["lowpass"] * 1.25, Nyquist)
    _check_option("normalization", normalization, ["length", "full"])
    psds, freq = psd_multitaper(
        epochs_src,
        fmin=fmin,
        fmax=fmax,
        tmin=tmin,
        tmax=tmax,
        bandwidth=bandwidth,
        adaptive=adaptive,
        low_bias=low_bias,
        normalization=normalization,
        picks=picks,
        proj=proj,
        n_jobs=n_jobs,
        reject_by_annotation=reject_by_annotation,
        verbose=None,
    )
    if dB:
        np.log10(np.maximum(psds, np.finfo(float).tiny), out=psds)
        psds *= 10
        psds_mean = psds.mean(axis=0)
    else:
        psds_mean = psds.mean(axis=0)
    return psds_mean  # psds_mean = (n_components, n_freqs)