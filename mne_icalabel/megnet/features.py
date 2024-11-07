import io

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.io import BaseRaw
from mne.preprocessing import ICA
from mne.utils import _validate_type, warn
from numpy.typing import NDArray
from PIL import Image
from scipy import interpolate
from scipy.spatial import ConvexHull

from ._utils import cart2sph, pol2cart


def get_megnet_features(raw: BaseRaw, ica: ICA):
    """Extract time series and topomaps for each ICA component.

    MEGNet uses topomaps from BrainStorm exported as 120x120x3 RGB images. Thus, we need
    to replicate the 'appearance'/'look' of a BrainStorm topomap.

    Parameters
    ----------
    raw : Raw.
        Raw MEG recording used to fit the ICA decomposition. The raw instance should be
        bandpass filtered between 1 and 100 Hz and notch filtered at 50 or 60 Hz to
        remove line noise, and downsampled to 250 Hz.
    ica : ICA
        ICA decomposition of the provided instance. The ICA decomposition
        should use the infomax method.

    Returns
    -------
    time_series : array of shape (n_components, n_samples)
        The time series for each ICA component.
    topomaps : array of shape (n_components, 120, 120, 3)
        The topomap RGB images for each ICA component.

    """
    _validate_type(raw, BaseRaw, "raw")
    _validate_type(ica, ICA, "ica")

    if not any(
        ch_type in ["mag", "grad"] for ch_type in raw.get_channel_types(unique=True)
    ):
        raise RuntimeError(
            "Could not find MEG channels in the provided Raw instance. The MEGnet "
            "model was fitted on MEG data and is not suited for other types of "
            "channels."
        )

    if n_samples := raw.get_data().shape[1] < 15000:
        raise RuntimeError(
            f"The provided raw instance has {n_samples} points. MEGnet was designed to "
            "classify features extracted from an MEG dataset at least 60 seconds long "
            "@ 250 Hz, corresponding to at least. 15 000 samples."
        )

    if _check_notch(raw, "mag"):
        raise RuntimeError(
            "Line noise detected in 50/60 Hz. MEGnet was trained on MEG data without "
            "line noise. Please remove line noise before using MEGnet "
            "(see the 'notch_filter()' method for Raw instances."
        )

    if not np.isclose(raw.info["sfreq"], 250, atol=1e-1):
        warn(
            "The provided raw instance is not sampled at 250 Hz "
            f"(sfreq={raw.info['sfreq']} Hz). "
            "MEGnet was designed to classify features extracted from"
            "an MEG dataset sampled at 250 Hz "
            "(see the 'resample()' method for raw). "
            "The classification performance might be negatively impacted."
        )

    if raw.info["highpass"] != 1 or raw.info["lowpass"] != 100:
        warn(
            "The provided raw instance is not filtered between 1 and 100 Hz. "
            "MEGnet was designed to classify features extracted from an MEG dataset "
            "bandpass filtered between 1 and 100 Hz (see the 'filter()' method for "
            "Raw). The classification performance might be negatively impacted."
        )

    if ica.method != "infomax":
        warn(
            f"The provided ICA instance was fitted with a '{ica.method}' algorithm. "
            "MEGnet was designed with infomax ICA decompositions. To use the "
            "infomax algorithm, use mne.preprocessing.ICA instance with "
            "the arguments ICA(method='infomax')."
        )

    pos_new, outlines = _get_topomaps_data(ica)
    topomaps = _get_topomaps(ica, pos_new, outlines)
    time_series = ica.get_sources(raw).get_data()

    return time_series, topomaps


def _make_head_outlines(sphere: NDArray, pos: NDArray, clip_origin: tuple):
    """Generate head outlines and mask positions for the topomap plot."""
    x, y, _, radius = sphere
    ll = np.linspace(0, 2 * np.pi, 101)
    head_x = np.cos(ll) * radius * 1.01 + x
    head_y = np.sin(ll) * radius * 1.01 + y

    mask_scale = max(1.0, np.linalg.norm(pos, axis=1).max() * 1.01 / radius)
    clip_radius = radius * mask_scale

    outlines_dict = {
        "head": (head_x, head_y),
        "mask_pos": (mask_scale * head_x, mask_scale * head_y),
        "clip_radius": (clip_radius,) * 2,
        "clip_origin": clip_origin,
    }
    return outlines_dict


def _get_topomaps_data(ica: ICA):
    """Prepare 2D sensor positions and outlines for topomap plotting."""
    mags = mne.pick_types(ica.info, meg="mag")
    channel_info = ica.info["chs"]
    loc_3d = [channel_info[i]["loc"][0:3] for i in mags]
    channel_locations_3d = np.array(loc_3d)

    # Convert to spherical and then to 2D
    sph_coords = np.transpose(
        cart2sph(
            channel_locations_3d[:, 0],
            channel_locations_3d[:, 1],
            channel_locations_3d[:, 2],
        )
    )
    TH, PHI = sph_coords[:, 1], sph_coords[:, 2]
    newR = 1 - PHI / np.pi * 2
    channel_locations_2d = np.transpose(pol2cart(newR, TH))

    # Adjust coordinates with convex hull interpolation
    hull = ConvexHull(channel_locations_2d)
    border_indices = hull.vertices
    Dborder = 1 / newR[border_indices]

    funcTh = np.hstack(
        [
            TH[border_indices] - 2 * np.pi,
            TH[border_indices],
            TH[border_indices] + 2 * np.pi,
        ]
    )
    funcD = np.hstack((Dborder, Dborder, Dborder))
    interp_func = interpolate.interp1d(funcTh, funcD)
    D = interp_func(TH)

    adjusted_R = np.array([min(newR[i] * D[i], 1) for i in range(len(mags))])
    Xnew, Ynew = pol2cart(adjusted_R, TH)
    pos_new = np.vstack((Xnew, Ynew)).T

    outlines = _make_head_outlines(np.array([0, 0, 0, 1]), pos_new, (0, 0))
    return pos_new, outlines


def _get_topomaps(ica: ICA, pos_new: NDArray, outlines: dict):
    """Generate topomap images for each ICA component."""
    topomaps = []
    data_picks = mne.pick_types(ica.info, meg="mag")
    components = ica.get_components()

    for comp in range(ica.n_components_):
        data = components[data_picks, comp]
        fig = plt.figure(figsize=(1.3, 1.3), dpi=100, facecolor="black")
        ax = fig.add_subplot(111)
        mnefig, _ = mne.viz.plot_topomap(
            data,
            pos_new,
            sensors=False,
            outlines=outlines,
            extrapolate="head",
            sphere=[0, 0, 0, 1],
            contours=0,
            res=120,
            axes=ax,
            show=False,
            cmap="bwr",
        )
        img_buf = io.BytesIO()
        mnefig.figure.savefig(
            img_buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0
        )
        img_buf.seek(0)
        rgba_image = Image.open(img_buf)
        rgb_image = rgba_image.convert("RGB")
        img_buf.close()
        plt.close(fig)

        topomaps.append(np.array(rgb_image))

    return np.array(topomaps)


def _line_noise_channel(
    raw: BaseRaw,
    picks: str,
    fline: float = 50.0,
    neighbor_width: float = 2.0,
    threshold_factor: float = 3,
    show: bool = False,
):
    """Detect line noise in MEG/EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw MEG/EEG data.
    picks : str or list, optional
        Channels to include in the analysis.
    fline : float, optional
        The base frequency of the line noise to detect.
    neighbor_width : float, optional
        Width of the frequency neighborhood around each harmonic
        for calculating background noise, in Hz. Default is 2.0 Hz.
    threshold_factor : float, optional
        Multiplicative factor for setting the detection threshold.
        The threshold is set as the mean of the neighboring frequencies plus
        `threshold_factor` times the standard deviation. Default is 3.
    show : bool, optional
        Whether to plot the PSD for channels affected by line noise.

    Returns
    -------
    bool
        Returns True if line noise is detected in any channel, otherwise False.

    """
    psd = raw.compute_psd(picks=picks)
    freqs = psd.freqs
    psds = psd.get_data()
    ch_names = psd.ch_names

    # Compute Nyquist frequency and determine maximum harmonic
    nyquist_freq = raw.info["sfreq"] / 2.0
    max_harmonic = int(nyquist_freq // fline)

    # Generate list of harmonic frequencies based on the fundamental frequency
    line_freqs = np.arange(fline, fline * (max_harmonic + 1), fline)
    freq_res = freqs[1] - freqs[0]
    n_neighbors = int(np.ceil(neighbor_width / freq_res))

    line_noise_detected = []
    for ch_idx in range(psds.shape[0]):
        psd_ch = psds[ch_idx, :]  # PSD for the current channel
        for lf in line_freqs:
            # Find the frequency index closest to the current harmonic
            idx = np.argmin(np.abs(freqs - lf))
            # Get index range for neighboring frequencies,
            # excluding the harmonic frequency itself
            idx_range = np.arange(
                max(0, idx - n_neighbors), min(len(freqs), idx + n_neighbors + 1)
            )
            idx_neighbors = idx_range[idx_range != idx]
            # Calculate mean and standard deviation of neighboring frequencies
            neighbor_mean = np.mean(psd_ch[idx_neighbors])
            neighbor_std = np.std(psd_ch[idx_neighbors])

            threshold = neighbor_mean + threshold_factor * neighbor_std
            if psd_ch[idx] > threshold:
                line_noise_detected.append(
                    {"channel": ch_names[ch_idx], "frequency": lf}
                )

    if show and line_noise_detected:
        affected_channels = set([item["channel"] for item in line_noise_detected])
        plt.figure(figsize=(12, 3))
        for ch_name in affected_channels:
            ch_idx = ch_names.index(ch_name)
            plt.semilogy(freqs, psds[ch_idx, :], label=ch_name)
        plt.axvline(fline, color="k", linestyle="--", lw=3, alpha=0.3)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (PSD)")
        plt.legend()
        plt.show()

    return line_noise_detected


def _check_notch(
    raw: BaseRaw, picks: str, neighbor_width: float = 2.0, threshold_factor: float = 3
) -> bool:
    """Return True if line noise find in raw."""
    check_result = False
    for fline in [50, 60]:
        if _line_noise_channel(raw, picks, fline, neighbor_width, threshold_factor):
            check_result = True
            break
    return check_result
