from _typeshed import Incomplete
from mne import Info
from mne.preprocessing import ICA
from numpy.typing import NDArray

from ..utils._checks import _validate_ica as _validate_ica
from ..utils._docs import fill_doc as fill_doc

def get_topomaps(
    ica: ICA,
    picks: Incomplete | None = None,
    res: int = 64,
    image_interp: str = ...,
    border: float | str = ...,
    extrapolate: str = ...,
) -> dict[str, NDArray[float]]:
    """Generate an array of scalp topographies for the picked components.

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
    res : int
        The resolution of the square topographic map (in pixels).
    image_interp : str
        The image interpolation to be used. All matplotlib options are
        accepted.
    border : float | 'mean'
        Value to extrapolate to on the topomap borders. If ``'mean'`` (default),
        then each extrapolated point has the average value of its neighbours.
    extrapolate : str
        Options:

        - ``'box'``
            Extrapolate to four points placed to form a square encompassing all
            data points, where each side of the square is three times the range
            of the data in the respective dimension.
        - ``'local'`` (default for MEG sensors)
            Extrapolate only to nearby points (approximately to points closer than
            median inter-electrode distance). This will also set the
            mask to be polygonal based on the convex hull of the sensors.
        - ``'head'`` (default for non-MEG sensors)
            Extrapolate out to the edges of the clipping circle. This will be on
            the head circle when the sensors are contained within the head circle,
            but it can extend beyond the head when sensors are plotted outside
            the head circle.

    Returns
    -------
    topomaps : dict of array of shape (n_components, n_pixels, n_pixels)
        Dictionary of ICs topographic maps for each channel type.
    """

def _get_topomap_array(
    data: NDArray[float],
    info: Info,
    res: int = 64,
    image_interp: str = ...,
    border: float | str = ...,
    extrapolate: str = ...,
) -> NDArray[float]:
    """Generate a scalp topographic map (n_pixels, n_pixels).

    Parameters
    ----------
    data : array of shape (n_channels,)
        The data points used to generate the topographic map.
    info : Info
        Instance of `mne.Info` with the montage associated with the ``(n_channels,)``
        points.
    res : int
        The resolution of the square topographic map (in pixels).
    image_interp : str
        The image interpolation to be used. All matplotlib options are
        accepted.
    border : float | 'mean'
        Value to extrapolate to on the topomap borders. If ``'mean'`` (default),
        then each extrapolated point has the average value of its neighbours.
    extrapolate : str
        Options:

        - ``'box'``
            Extrapolate to four points placed to form a square encompassing all
            data points, where each side of the square is three times the range
            of the data in the respective dimension.
        - ``'local'`` (default for MEG sensors)
            Extrapolate only to nearby points (approximately to points closer than
            median inter-electrode distance). This will also set the
            mask to be polygonal based on the convex hull of the sensors.
        - ``'head'`` (default for non-MEG sensors)
            Extrapolate out to the edges of the clipping circle. This will be on
            the head circle when the sensors are contained within the head circle,
            but it can extend beyond the head when sensors are plotted outside
            the head circle.

    Returns
    -------
    topomap : array of shape (n_pixels, n_pixels)
        Topographic map array.
    """
