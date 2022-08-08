from typing import Optional, Union

import numpy as np
from mne.channels.layout import _find_topomap_coords
from mne.defaults import _BORDER_DEFAULT, _EXTRAPOLATE_DEFAULT, _INTERPOLATION_DEFAULT
from mne.io import Info
from mne.io.pick import _get_channel_types, _pick_data_channels, _picks_to_idx, pick_info
from mne.preprocessing import ICA
from mne.viz.topomap import _check_extrapolate, _make_head_outlines, _setup_interp
from numpy.typing import NDArray

from ..utils._docs import fill_doc


@fill_doc
def get_topomaps(
    ica: ICA,
    picks=None,
    res: int = 64,
    outlines: Optional[str] = "head",
    image_interp: str = _INTERPOLATION_DEFAULT,  # 'cubic'
    border: Union[float, str] = _BORDER_DEFAULT,  # 'mean'
    extrapolate: str = _EXTRAPOLATE_DEFAULT,  # 'head'
):
    """Generate an array of scalp topographies (n_pixels, n_pixels) for the picked components.

    Parameters
    ----------
    ica : ICA
        MNE `~mne.preprocessing.ICA` decomposition.
    %(picks_ica)s ``None`` (default) will pick all independent components in the order fitted.
    %(res_topomap)s
    %(outlines_topomap)s
    %(image_interp_topomap)s
    %(border_topomap)s
    %(extrapolate_topomap)s

    Returns
    -------
    topomaps : array of shape (n_components, n_pixels, n_pixels)
        Topographic maps of each picked independent component.
    """
    picks = _picks_to_idx(ica.n_components_, picks)
    data = np.dot(
        ica.mixing_matrix_[:, : ica.n_components_].T,
        ica.pca_components_[: ica.n_components_],
    )
    # Create an empty array of size (len(picks), 64, 64) for the topomap
    topomaps = np.zeros((len(picks), res, res))
    for i, component in enumerate(picks):
        topo = np.flipud(
            _get_topomap_array(
                data[component, :], ica.info, res, outlines, image_interp, border, extrapolate
            )
        )
        # Set NaN values to 0
        np.nan_to_num(topo, nan=0.0, copy=False)
        # Standardize the values
        topomaps[i, :, :] = topo / np.max(np.abs(topo))
    return topomaps  # topographic map array for all the picked components (len(picks), 64, 64)


@fill_doc
def _get_topomap_array(
    data: NDArray[float],
    pos: Info,
    res: int = 64,
    outlines: Optional[str] = "head",
    image_interp: str = _INTERPOLATION_DEFAULT,  # 'cubic'
    border: Union[float, str] = _BORDER_DEFAULT,  # 'mean'
    extrapolate: str = _EXTRAPOLATE_DEFAULT,  # 'head'
):
    """Generate a scalp topographic map (n_pixels, n_pixels).

    Parameters
    ----------
    data : array of shape (n_channels,)
        The data points used to generate the topographic map.
    pos : Info
        Instance of `mne.Info` with the montage associated with the ``(n_channels,)`` points.
    %(res_topomap)s
    %(outlines_topomap)s
    %(image_interp_topomap)s
    %(border_topomap)s
    %(extrapolate_topomap)s

    Returns
    -------
    topomap : array of shape (n_pixels, n_pixels)
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

    # interpolation, valid only for MNE ≥ 1.1
    outlines = _make_head_outlines(sphere, pos, outlines, (0.0, 0.0))
    extent, Xi, Yi, interp = _setup_interp(pos, res, image_interp, extrapolate, outlines, border)
    interp.set_values(data)
    return interp.set_locations(Xi, Yi)()  # Zi, topomap of shape (n_pixels, n_pixels)
