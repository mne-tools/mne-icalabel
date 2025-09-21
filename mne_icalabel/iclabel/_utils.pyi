from mne.io import BaseRaw
from numpy.typing import ArrayLike, NDArray

def _mne_to_eeglab_locs(
    raw: BaseRaw, picks: list[str]
) -> tuple[NDArray[float], NDArray[float]]:
    """Obtain EEGLab-like spherical coordinate from EEG channel positions.

    TODO: @JACOB:
    - Where is (0,0,0) defined in MNE vs EEGLab?
    - some text description of how the sphere coordinates differ between MNE
    and EEGLab.

    Parameters
    ----------
    raw : mne.io.BaseRaw
    Instance of raw object with a `mne.montage.DigMontage` set with
    ``n_channels`` channel positions.
    picks : list of str
    List of channel names to include.

    Returns
    -------
    Rd : np.array of shape (1, n_channels)
    Angle in spherical coordinates of each EEG channel.
    Th : np.array of shape (1, n_channels)
    Degree in spherical coordinates of each EEG channel.
    """

def _next_power_of_2(x) -> int:
    """Equivalent to 2^nextpow2 in MATLAB."""

def _gdatav4(
    x: ArrayLike, y: ArrayLike, v: ArrayLike, xq: ArrayLike, yq: ArrayLike
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """GDATAV4 MATLAB 4 GRIDDATA interpolation.

    Parameters
    ----------
    x : array
    x-coordinates
    y : array
    y-coordinates
    v : array
    values
    xq : array
    x-grid
    yq : array
    y-grid

    Returns
    -------
    xq : array
    yq : array
    vq : array

    Reference
    ---------
    David T. Sandwell, Biharmonic spline interpolation of GEOS-3 and SEASAT
    altimeter data, Geophysical Research Letters, 2, 139-142, 1987.

    Describes interpolation using value of gradient of value in any dimension.
    """

def _mergepoints2D(
    x: ArrayLike, y: ArrayLike, v: ArrayLike
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Averages values for points that are close to each other.

    Parameters
    ----------
    x : array
    x-coordinates
    y : array
    y-coordinates
    v : array
    values

    Returns
    -------
    x : array
    y : array
    v : array
    """

def _mergesimpts(
    data: ArrayLike, tols: list[ArrayLike], mode: str = "average"
) -> ArrayLike:
    """Merge similar points."""
