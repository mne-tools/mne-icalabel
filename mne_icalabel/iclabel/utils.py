import warnings
from typing import List, Tuple

import numpy as np
from mne.io import BaseRaw
from numpy.typing import ArrayLike, NDArray


def _mne_to_eeglab_locs(raw: BaseRaw, picks: List[str]) -> Tuple[NDArray[float], NDArray[float]]:
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

    def _sph2topo(_theta, _phi):
        """Convert spherical coordinates to topo."""
        az = _phi
        horiz = _theta
        angle = -1 * horiz
        radius = (np.pi / 2 - az) / np.pi
        return angle, radius

    def _cart2sph(_x, _y, _z):
        """Convert cartesian coordinates to spherical."""
        azimuth = np.arctan2(_y, _x)
        elevation = np.arctan2(_z, np.sqrt(_x**2 + _y**2))
        r = np.sqrt(_x**2 + _y**2 + _z**2)
        # theta,phi,r
        return azimuth, elevation, r

    # get the channel position dictionary
    montage = raw.copy().pick_channels(picks, ordered=True).get_montage()
    positions = montage.get_positions()
    ch_pos = positions["ch_pos"]

    # get locations as a 2D array
    locs = np.vstack(list(ch_pos.values()))

    # Obtain cartesian coordinates
    x = locs[:, 1]

    # be mindful of the nose orientation in eeglab and mne
    # TODO: @Jacob, please expand on this.
    y = -1 * locs[:, 0]
    # see https://github.com/mne-tools/mne-python/blob/24377ad3200b6099ed47576e9cf8b27578d571ef/mne/io/eeglab/eeglab.py#L105  # noqa
    z = locs[:, 2]

    # Obtain Spherical Coordinates
    sph = np.array([_cart2sph(x[i], y[i], z[i]) for i in range(len(x))])
    theta = sph[:, 0]
    phi = sph[:, 1]

    # Obtain Polar coordinates (as in eeglab)
    topo = np.array([_sph2topo(theta[i], phi[i]) for i in range(len(theta))])
    rd = topo[:, 1]
    th = topo[:, 0]

    return rd.reshape([1, -1]), np.degrees(th).reshape([1, -1])


def _pol2cart(theta: NDArray[float], rho: NDArray[float]) -> Tuple[NDArray[float], NDArray[float]]:
    """Convert polar coordinates to cartesian coordinates.

    Parameters
    ----------
    theta : array
        angle
    rho : array
        magnitude
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


# ----------------------------------------------------------------------------
def _next_power_of_2(x) -> int:
    """Equivalent to 2^nextpow2 in MATLAB."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


# ----------------------------------------------------------------------------
def _gdatav4(
    x: ArrayLike, y: ArrayLike, v: ArrayLike, xq: ArrayLike, yq: ArrayLike
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
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
    x, y, v = _mergepoints2D(x, y, v)

    xy = x + 1j * y
    xy = np.squeeze(xy)

    # Determine distances between points
    d = np.abs(np.subtract.outer(xy, xy))
    # Determine weights for interpolation
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="divide by zero encountered in log", category=RuntimeWarning
        )
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in multiply", category=RuntimeWarning
        )
        g = np.square(d) * (np.log(d) - 1)  # Green's function.
    # Fixup value of Green's function along diagonal
    np.fill_diagonal(g, 0)
    weights = np.linalg.lstsq(g, v, rcond=-1)[0]

    m, n = xq.shape
    vq = np.zeros(xq.shape)

    # Evaluate at requested points (xq,yq). Loop to save memory.
    for i in range(m):
        for j in range(n):
            d = np.abs(xq[i, j] + 1j * yq[i, j] - xy)
            g = np.square(d) * (np.log(d) - 1)
            # Value of Green's function at zero
            g[np.where(np.isclose(d, 0))] = 0
            vq[i, j] = (np.expand_dims(g, axis=0) @ np.expand_dims(weights, axis=1))[0][0]
    return xq, yq, vq


def _mergepoints2D(
    x: ArrayLike, y: ArrayLike, v: ArrayLike
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
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
    # Sort x and y so duplicate points can be averaged
    # Need x,y and z to be column vectors
    sz = x.size
    x = x.copy()
    y = y.copy()
    v = v.copy()
    x = np.reshape(x, sz, order="F")
    y = np.reshape(y, sz, order="F")
    v = np.reshape(v, sz, order="F")

    myepsx = np.spacing(0.5 * (np.max(x) - np.min(x))) ** (1 / 3)
    myepsy = np.spacing(0.5 * (np.max(y) - np.min(y))) ** (1 / 3)
    # Look for x, y points that are identical (within a tolerance)
    # Average out the values for these points
    if np.all(np.isreal(v)):
        data = np.stack((y, x, v), axis=-1)
        yxv = _mergesimpts(data, [myepsy, myepsx, np.inf], "average")
        x = yxv[:, 1]
        y = yxv[:, 0]
        v = yxv[:, 2]
    else:
        # If z is imaginary split out the real and imaginary parts
        data = np.stack((y, x, np.real(v), np.imag(v)), axis=-1)
        yxv = _mergesimpts(data, [myepsy, myepsx, np.inf, np.inf], "average")
        x = yxv[:, 1]
        y = yxv[:, 0]
        # Re-combine the real and imaginary parts
        v = yxv[:, 2] + 1j * yxv[:, 3]

    return x, y, v


def _mergesimpts(
    data: ArrayLike, tols: List[ArrayLike], mode: str = "average"
) -> ArrayLike:  # noqa
    """
    Parameters
    ----------
    data : array
    tols : list of 3 arrays
    mode : str

    Returns
    -------
    array
    """
    data_ = data.copy()[np.argsort(data[:, 0])]
    newdata = []
    tols_ = np.array(tols)
    idxs_ready: List[int] = []
    point = 0
    for point in range(data_.shape[0]):
        if point in idxs_ready:
            continue
        else:
            similar_pts = np.where(np.prod(np.abs(data_ - data_[point]) < tols_, axis=-1))
            similar_pts = np.array(list(set(similar_pts[0].tolist()) - set(idxs_ready)))
            idxs_ready += similar_pts.tolist()
            if mode == "average":
                exemplar = np.mean(data_[similar_pts], axis=0)
            else:
                exemplar = data_[similar_pts].copy()[0]  # first
            newdata.append(exemplar)
    return np.array(newdata)
