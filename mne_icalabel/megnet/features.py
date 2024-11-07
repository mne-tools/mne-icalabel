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
    if _check_line_noise(raw):
        warn(
            "Line noise detected in 50/60 Hz. MEGnet was trained on MEG data without "
            "line noise. Please remove line noise before using MEGnet "
            "(see the 'notch_filter()' method for Raw instances."
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


def _check_line_noise(
    raw: BaseRaw, *, neighbor_width: int = 4, threshold_factor: int = 10
) -> bool:
    """Check if line noise is present in the MEG/EEG data."""
    if raw.info.get("line_freq", None) is None:  # we don't know the line frequency
        return False
    # validate the primary and first harmonic frequencies
    nyquist_freq = raw.info["sfreq"] / 2.0
    line_freqs = [raw.info["line_freq"], 2 * raw.info["line_freq"]]
    if any(nyquist_freq < lf for lf in line_freqs):
        # not raising because if we get here, it means that someone provided a raw with
        # a sampling rate extremely low (100 Hz?) and (1) either they missed all
        # of the previous warnings encountered or (2) they know what they are doing.
        warn("The sampling rate raw.info['sfreq'] is too low to estimate line niose.")
        return False
    # compute the power spectrum and retrieve the frequencies of interest
    spectrum = raw.compute_psd(picks="meg", exclude="bads")
    data, freqs = spectrum.get_data(
        fmin=raw.info["line_freq"] - neighbor_width,
        fmax=raw.info["line_freq"] + neighbor_width,
        return_freqs=True,
    )  # array of shape (n_good_channel, n_freqs)
    idx = np.argmin(np.abs(freqs - raw.info["line_freq"]))
    mask = np.ones(data.shape[1], dtype=bool)
    mask[idx] = False
    background_mean = np.mean(data[:, mask], axis=1)
    background_std = np.std(data[:, mask], axis=1)
    threshold = background_mean + threshold_factor * background_std
    return np.any(data[:, idx] > threshold)
