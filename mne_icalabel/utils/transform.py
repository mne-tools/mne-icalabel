from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def pol2cart(theta: NDArray, rho: NDArray) -> tuple[NDArray, NDArray]:
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


def sph2topo(theta, phi):
    """Convert spherical coordinates to topo."""
    az = phi
    horiz = theta
    angle = -1 * horiz
    radius = (np.pi / 2 - az) / np.pi
    return angle, radius


def cart2sph(x, y, z):
    """Convert cartesian coordinates to spherical."""
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    # r, theta, phi
    return r, azimuth, elevation
