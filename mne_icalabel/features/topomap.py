import numpy as np
from mne.io.pick import (
    pick_info,
    _pick_data_channels,
    _get_channel_types,
)
from mne.channels.layout import _find_topomap_coords
from mne.viz.topomap import _check_extrapolate, _make_head_outlines, _setup_interp
from mne.viz.utils import _setup_vmin_vmax
from mne.defaults import _INTERPOLATION_DEFAULT, _EXTRAPOLATE_DEFAULT, _BORDER_DEFAULT
from mne.io import Info


def get_topo_array(ica, picks="eeg"):

    """Generates a array of scalp topographic plot of size 64*64

    Parameters:
    ----------

    ica: ICA
        Instance of MNE ICA Decomposition mne.preprocesssing.ICA

    picks: str | list | slice | None
    In lists, channel type strings (e.g., ['meg', 'eeg']) will pick channels
    of those types, channel name strings (e.g., ['MEG0111', 'MEG2623'] will
    pick the given channels. Can also be the string values “all” to pick all
    channels, or “data” to pick data channels. None (default) will pick all
    channels. Note that channels in info['bads'] will be included if their
    names or indices are explicitly provided.

    Returns:
    --------
    topo_array: np.ndarray of shape (n_components, 64,64)

    """

    n_components = ica.mixing_matrix_.shape[1]

    data = np.dot(
        ica.mixing_matrix_[:, : ica.n_components_].T,
        ica.pca_components_[: ica.n_components_],
    )

    # Create an empty array of size (n_components, 64, 64) to fit topo values
    topo_array = np.zeros((n_components, 64, 64))

    # f, ax = plt.subplots(1, ica.n_components_) #For Visualization

    for j in range(n_components):
        topo_ = np.flipud(topographic_map(data[j, :], ica.info))

        # ax[j].imshow(topo_) # For visualization

        # Set NaN values to 0
        np.nan_to_num(topo_, copy=False)

        # Standardize the values
        topo_array[j, :, :] = topo_ / np.max(np.abs(topo_))

    return topo_array  # topographic map array for all the components (n_components, 64, 64)


def topographic_map(
    data,
    pos: Info,
    vmin=None,
    vmax=None,
    res=64,
    outlines="head",
    image_interp=_INTERPOLATION_DEFAULT,  #'cubic
    border=_BORDER_DEFAULT,  #'mean'
    extrapolate=_EXTRAPOLATE_DEFAULT,  #'head'
):
    """Generates a topographic map as image

    Parameters:
    -----------

    data: array of shape (n_channels,)
        The data values to plot.

    pos: instance of mne.info

    vmin: float | callable | None
        The value specifying the lower bound of the color range.

    vmax: float | callable | None
        The value specifying the upper bound of the color range.

    res: int
        The resolution of the topomap image (n pixels in eaxh side, for example 64*64 in this case)

    %(outlines_topomap)s

    image_interp: str
        The image interpolation to be used. All matplotlib options are
        accepted.

    %(border_topomap)s

    %(extrapolate_topomap)s


    Returns:
    --------
    Zi: Array of size (64*64)
        # Topographic map array

    """
    picks = _pick_data_channels(pos, exclude=())  # pick only data channels
    pos = pick_info(pos, picks)
    ch_type = _get_channel_types(pos, unique=True)

    if len(ch_type) > 1:
        raise ValueError("Multiple channel types in Info structure.")
    elif len(pos["chs"]) != data.shape[0]:
        raise ValueError("Number of channels in the Info object the data array do not match.")
    else:
        ch_type = ch_type.pop()

    picks = list(range(data.shape[0]))
    sphere = np.array([0.0, 0.0, 0.0, 0.095])

    # inferring x,y coordinates form mne.info instance
    pos = _find_topomap_coords(pos, picks=picks, sphere=sphere)
    extrapolate = _check_extrapolate(extrapolate, ch_type)

    if len(data) != len(pos):
        raise ValueError(
            "Data and pos need to be of same length. Got data of "
            "length %s, pos of length %s" % (len(data), len(pos))
        )

    norm = min(data) >= 0
    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax, norm)

    # interpolation
    outlines = _make_head_outlines(sphere, pos, outlines, (0.0, 0.0))
    extent, Xi, Yi, interp = _setup_interp(pos, res, image_interp, extrapolate, outlines, border)
    interp.set_values(data)
    Zi = interp.set_locations(Xi, Yi)()
    return Zi  # Topographic map array of size (64*64)
