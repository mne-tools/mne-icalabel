import numpy as np
from mne.channels.layout import _find_topomap_coords
from mne.defaults import _BORDER_DEFAULT, _EXTRAPOLATE_DEFAULT, _INTERPOLATION_DEFAULT
from mne.io import Info
from mne.io.pick import (
    _get_channel_types,
    _pick_data_channels,
    _picks_to_idx,
    pick_info,
)
from mne.preprocessing import ICA
from mne.viz.topomap import _check_extrapolate, _make_head_outlines, _setup_interp
from mne.viz.utils import _setup_vmin_vmax
from numpy.typing import NDArray


def get_topomaps(ica: ICA, picks=None):
    """Generate an array of scalp topographies (n_pixels, n_pixels) for the picked components.

    Parameters
    ----------
    ica : ICA
        Instance of MNE `~mne.preprocessing.ICA` decomposition.
    %(picks_ica)s ``None`` (default) will pick all sources in the order fitted.

    Returns
    -------
    topomaps : array of shape (n_components, n_pixels, n_pixels)
    """
    if picks is None:  # plot all components
        picks = range(0, ica.n_components_)
    else:
        picks = _picks_to_idx(ica.info, picks)
    data = np.dot(
        ica.mixing_matrix_[:, : ica.n_components_].T,
        ica.pca_components_[: ica.n_components_],
    )
    # Create an empty array of size (n_components, 64, 64) to fit topo values
    topo_array = np.zeros((len(picks), 64, 64))

    for j in range(len(picks)):
        topo_ = np.flipud(get_topomap(data[j, :], ica.info))
        # Set NaN values to 0
        np.nan_to_num(topo_, nan=0.0, copy=False)
        # Standardize the values
        topo_array[j, :, :] = topo_ / np.max(np.abs(topo_))

    return topo_array  # topographic map array for all the components (n_components, 64, 64)


def get_topomap(
    data: NDArray[float],
    pos: Info,
    vmin=None,
    vmax=None,
    res: int = 64,
    outlines="head",
    image_interp=_INTERPOLATION_DEFAULT,  # 'cubic'
    border=_BORDER_DEFAULT,  # 'mean'
    extrapolate=_EXTRAPOLATE_DEFAULT,  # 'head'
):
    """Generate a scalp topographic map (n_pixels, n_pixels).

    Parameters
    ----------
    data : array of shape (n_channels,)
        The data points used to generate the topographic map.
    pos : `mne.Info`
        Instance of `mne.Info` with the montage associated with the (n_channels,) points.
    vmin : float | callable | None
        The value specifying the lower bound of the color range.
    vmax : float | callable | None
        The value specifying the upper bound of the color range.
    res : int = 64
        The resolution of the square topographic map (in pixels).
    %(outlines_topomap)s
    image_interp : str
        The image interpolation to be used. All matplotlib options are
        accepted.
    %(border_topomap)s
    %(extrapolate_topomap)s

    Returns
    -------
    topomap : array of size (n_pixels, n_pixels)
        Topographic map array.
    """
    picks = _pick_data_channels(pos, exclude=())  # pick only data channels
    pos = pick_info(pos, picks)
    ch_type = _get_channel_types(pos, unique=True)

    if len(ch_type) > 1:
        raise ValueError("Multiple channel types in Info structure.")
    elif len(pos["chs"]) != data.shape[0]:
        raise ValueError(
            "The number of channels in the Info object and in the data array do not match."
        )
    else:
        ch_type = ch_type.pop()

    picks = list(range(data.shape[0]))
    sphere = np.array([0.0, 0.0, 0.0, 0.095])

    # inferring (x, y) coordinates form mne.Info instance
    pos = _find_topomap_coords(pos, picks=picks, sphere=sphere)
    extrapolate = _check_extrapolate(extrapolate, ch_type)
    norm = min(data) >= 0
    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax, norm)

    # interpolation
    outlines = _make_head_outlines(sphere, pos, outlines, (0.0, 0.0))
    extent, Xi, Yi, interp = _setup_interp(pos, res, image_interp, extrapolate, outlines, border)
    interp.set_values(data)
    return interp.set_locations(Xi, Yi)()  # Zi, topomap of shape (n_pixels, n_pixels)
