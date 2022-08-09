from typing import Dict, Union

import numpy as np
from mne.channels.layout import _find_topomap_coords
from mne.defaults import _BORDER_DEFAULT, _EXTRAPOLATE_DEFAULT, _INTERPOLATION_DEFAULT
from mne.io import Info
from mne.io.pick import _get_channel_types, _pick_data_channels, _picks_to_idx, pick_info
from mne.preprocessing import ICA
from mne.utils import _validate_type
from mne.viz.topomap import _check_extrapolate, _make_head_outlines, _setup_interp
from numpy.typing import NDArray

from ..utils._checks import _validate_ica
from ..utils._docs import fill_doc


@fill_doc
def get_topomaps(
    ica: ICA,
    picks=None,
    res: int = 64,
    image_interp: str = _INTERPOLATION_DEFAULT,  # 'cubic'
    border: Union[float, str] = _BORDER_DEFAULT,  # 'mean'
    extrapolate: str = _EXTRAPOLATE_DEFAULT,  # 'auto' -> 'head' for EEG, 'local' for MEG
) -> Dict[str, NDArray[float]]:
    """Generate an array of scalp topographies (n_pixels, n_pixels) for the picked components.

    Parameters
    ----------
    ica : ICA
        MNE `~mne.preprocessing.ICA` decomposition.
    picks : int | list of int | slice | None
        Indices of the independent components (ICs) to select.
        If an integer, represents the index of the IC to pick.
        Multiple ICs can be selected using a list of int or a slice.
        The indices are 0-indexed, so ``picks=1`` will pick the second IC: ``ICA001``.
        ``None`` (default) will pick all independent components in the order fitted.
    %(res_topomap)s
    %(image_interp_topomap)s
    %(border_topomap)s
    %(extrapolate_topomap)s

    Returns
    -------
    topomaps : dict of array of shape (n_components, n_pixels, n_pixels)
        Dictionary of ICs topographic maps for each channel type.
    """
    _validate_ica(ica)
    if isinstance(picks, str):
        raise TypeError(
            "Argument 'picks' should be an integer or a list of integers to select the ICs. "
            "Strings are not supported."
        )
    ic_picks = _picks_to_idx(ica.n_components_, picks)
    _validate_type(res, "int", "res", "int")
    if res <= 0:
        raise ValueError(
            f"Argument 'res' should be a strictly positive integer. Provided '{res}' is invalid."
        )
    # image_interp, border are validated by _setup_interp
    # extrapolate is validated by _check_extrapolate

    # prepare ICs
    data = np.dot(
        ica.mixing_matrix_.T,
        ica.pca_components_[: ica.n_components_],
    )
    # list channel types
    ch_picks = _pick_data_channels(ica.info, exclude=())
    ch_types = _get_channel_types(pick_info(ica.info, ch_picks), unique=True)

    # compute topomaps
    topomaps = dict()
    for ch_type in ch_types:
        topomaps[ch_type] = np.zeros((ic_picks.size, res, res))
        sel = _picks_to_idx(ica.info, picks=ch_type)
        info = pick_info(ica.info, sel)
        for k, component in enumerate(ic_picks):
            topomaps[ch_type][k, :, :] = _get_topomap_array(
                data[component, sel], info, res, image_interp, border, extrapolate
            )
    return topomaps


@fill_doc
def _get_topomap_array(
    data: NDArray[float],
    info: Info,
    res: int = 64,
    image_interp: str = _INTERPOLATION_DEFAULT,  # 'cubic'
    border: Union[float, str] = _BORDER_DEFAULT,  # 'mean'
    extrapolate: str = _EXTRAPOLATE_DEFAULT,  # 'auto' -> 'head' for EEG, 'local' for MEG
) -> NDArray[float]:
    """Generate a scalp topographic map (n_pixels, n_pixels).

    Parameters
    ----------
    data : array of shape (n_channels,)
        The data points used to generate the topographic map.
    info : Info
        Instance of `mne.Info` with the montage associated with the ``(n_channels,)`` points.
    %(res_topomap)s
    %(image_interp_topomap)s
    %(border_topomap)s
    %(extrapolate_topomap)s

    Returns
    -------
    topomap : array of shape (n_pixels, n_pixels)
        Topographic map array.
    """
    ch_type = _get_channel_types(info, unique=True)
    assert len(ch_type) == 1  # sanity-check
    ch_type = ch_type[0]
    picks = list(range(data.shape[0]))
    sphere = np.array([0.0, 0.0, 0.0, 0.095])

    # inferring (x, y) coordinates form mne.Info instance
    pos = _find_topomap_coords(info, picks=picks, sphere=sphere, ignore_overlap=True)
    extrapolate = _check_extrapolate(extrapolate, ch_type)

    # interpolation, valid only for MNE ≥ 1.1
    outlines = _make_head_outlines(sphere, pos, None, (0.0, 0.0))
    extent, Xi, Yi, interp = _setup_interp(pos, res, image_interp, extrapolate, outlines, border)
    interp.set_values(data)
    topomap = np.flipud(interp.set_locations(Xi, Yi)())  # Zi, shape (n_pixels, n_pixels)
    np.nan_to_num(topomap, nan=0.0, copy=False)
    topomap = topomap / np.max(np.abs(topomap))  # standardize
    return topomap  # (n_pixels, n_pixels)
