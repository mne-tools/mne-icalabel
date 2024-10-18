import io

import matplotlib.pyplot as plt
import mne  # type: ignore
import numpy as np
from _utils import cart2sph, pol2cart
from mne.io import BaseRaw  # type: ignore
from mne.preprocessing import ICA  # type: ignore
from mne.utils import warn  # type: ignore
from numpy.typing import NDArray
from PIL import Image
from scipy import interpolate  # type: ignore
from scipy.spatial import ConvexHull  # type: ignore
from ._utils import cart2sph, pol2cart


def get_megnet_features(raw: BaseRaw, ica: ICA):
    """
    Extract time series and topomaps for each ICA component.
    the main work is focused on making BrainStorm-like topomaps
    which trained the MEGnet.

    Parameters
    ----------
    raw : BaseRaw
        The raw MEG data. The raw instance should have 250 Hz
        sampling frequency and more than 60 seconds.
    ica : ICA
        The ICA object containing the independent components.

    Returns
    -------
    time_series : np.ndarray
        The time series for each ICA component.
    topomaps : np.ndarray
        The topomaps for each ICA component
    """
    if "meg" not in raw:
        raise RuntimeError(
            "Could not find MEG channels in the provided "
            "Raw instance. The MEGnet model was fitted on"
            "MEG data and is not suited for other types of channels."
        )

    if raw.times[-1] < 60:
        raise RuntimeError(
            f"The provided raw instance has {raw.times[-1]} seconds. "
            "MEGnet was designed to classify features extracted from "
            "an MEG datasetat least 60 seconds long. "
        )

    if not np.isclose(raw.info["sfreq"], 250, atol=1e-1):
        warn(
            "The provided raw instance is not sampled at 250 Hz"
            f"(sfreq={raw.info['sfreq']} Hz). "
            "MEGnet was designed to classify features extracted from"
            "an MEG dataset sampled at 250 Hz"
            "(see the 'resample()' method for raw)."
            "The classification performance might be negatively impacted."
        )

    pos_new, outlines = _get_topomaps_data(ica)
    topomaps = _get_topomaps(ica, pos_new, outlines)
    time_series = ica.get_sources(raw)._data

    return time_series, topomaps


def _make_head_outlines(sphere: NDArray, pos: NDArray, clip_origin: tuple):
    """
    Generate head outlines and mask positions for the topomap plot.
    """
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
    """
    Prepare 2D sensor positions and outlines for topomap plotting.
    """
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
    """
    Generate topomap images for each ICA component.
    """
    topomaps = []
    data_picks, _, _, _, _, _, _ = mne.viz.topomap._prepare_topomap_plot(
        ica, ch_type="mag"
    )
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
            contours=10,
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


if __name__ == "__main__":
    sample_dir = mne.datasets.sample.data_path()
    sample_fname = sample_dir / "MEG" / "sample" / "sample_audvis_raw.fif"

    raw = mne.io.read_raw_fif(sample_fname).pick_types("mag")
    raw.resample(250)
    ica = mne.preprocessing.ICA(n_components=20, method="infomax")
    ica.fit(raw)

    time_series, topomaps = get_megnet_features(raw, ica)

    fig, axes = plt.subplots(4, 5)
    for i, comp in enumerate(topomaps):
        row, col = divmod(i, 5)
        ax = axes[row, col]
        ax.imshow(comp)
        ax.axis("off")
    fig.tight_layout()
    plt.show()
