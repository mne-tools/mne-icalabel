import numpy as np


def pol2cart(theta: np.array, rho: np.array) -> tuple[np.array, np.array]:
    """
    Converts polar coordinates to cartesian coordinates.

    Args:
        theta (np.array): angle
        rho (np.array): magnitude
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def mergesimpts(data: np.array, tols: list[np.array, np.array, np.array], mode: str = 'average') -> np.array:
    """

    Args:
        data (np.array): [description]
        tols (list[np.array, np.array, np.array]): [description]
        mode (str, optional): [description]. Defaults to 'average'.

    Returns:
        np.array: [description]
    """
    data_ = data.copy()[np.argsort(data[:, 0])]
    newdata = []
    tols_ = np.array(tols)
    idxs_ready = []
    point = 0
    for point in range(data_.shape[0]):
        if point in idxs_ready:
            continue
        else:
            similar_pts = np.where(
                np.prod(np.abs(data_ - data_[point]) < tols_, axis=-1))
            similar_pts = np.array(
                list(set(similar_pts[0].tolist()) - set(idxs_ready)))
            idxs_ready += similar_pts.tolist()
            if mode == 'average':
                exemplar = np.mean(data_[similar_pts], axis=0)
            else:
                exemplar = data_[similar_pts].copy()[0]  # first
            newdata.append(exemplar)
    return np.array(newdata)


def mergepoints2D(x: np.array, y: np.array, v: np.array) -> tuple[np.array, np.array, np.array]:
    """
    Averages values for points that are close to each other.

    Args:
        x (np.array): x-coordinates
        y (np.array): y-coordinates
        v (np.array): values

    Returns:
        tuple[np.array, np.array, np.array]: [description]
    """
    # Sort x and y so duplicate points can be averaged
    # Need x,y and z to be column vectors
    sz = x.size
    x = x.copy()
    y = y.copy()
    v = v.copy()
    x = np.reshape(x, (sz), order='F')
    y = np.reshape(y, (sz), order='F')
    v = np.reshape(v, (sz), order='F')

    myepsx = np.spacing(0.5 * (np.max(x) - np.min(x)))**(1/3)
    myepsy = np.spacing(0.5 * (np.max(y) - np.min(y)))**(1/3)
    # Look for x, y points that are indentical (within a tolerance)
    # Average out the values for these points
    if np.all(np.isreal(v)):
        data = np.stack((y, x, v), axis=-1)
        yxv = mergesimpts(data, [myepsy, myepsx, np.inf], 'average')
        x = yxv[:, 1]
        y = yxv[:, 0]
        v = yxv[:, 2]
    else:
        # If z is imaginary split out the real and imaginary parts
        data = np.stack((y, x, np.real(v), np.imag(v)), axis=-1)
        yxv = mergesimpts(data, [myepsy, myepsx, np.inf, np.inf], 'average')
        x = yxv[:, 1]
        y = yxv[:, 0]
        # Re-combine the real and imaginary parts
        v = yxv[:, 2]+1j*yxv[:, 3]
    # Give a warning if some of the points were duplicates (and averaged out)
    # if sz > x.shape[0]:
    #     print('MATLAB:griddata:DuplicateDataPoints')
    return x, y, v


def gdatav4(x: np.array, y: np.array, v: np.array, xq: np.array, yq: np.array) -> tuple[np.array, np.array, np.array]:
    """
    GDATAV4 MATLAB 4 GRIDDATA interpolation
    Reference:  David T. Sandwell, Biharmonic spline
    interpolation of GEOS-3 and SEASAT altimeter
    data, Geophysical Research Letters, 2, 139-142,
    1987.  Describes interpolation using value or
    gradient of value in any dimension.

    Args:
        x (np.array): x-coordinates
        y (np.array): y-coordinates
        v (np.array): values
        xq (np.array): x-grid
        yq (np.array): y-grid

    Returns:
        tuple[np.array, np.array, np.array]: tuple of Xi, Yi, Zi 
    """

    x, y, v = mergepoints2D(x, y, v)

    xy = x + 1j*y
    xy = np.squeeze(xy)

    # Determine distances between points
    d = np.abs(np.subtract.outer(xy, xy))
    # % Determine weights for interpolation
    g = np.square(d) * (np.log(d)-1)  # % Green's function.
    # Fixup value of Green's function along diagonal
    np.fill_diagonal(g, 0)
    weights = np.linalg.lstsq(g, v)[0]

    m, n = xq.shape
    vq = np.zeros(xq.shape)

    # Evaluate at requested points (xq,yq).  Loop to save memory.
    for i in range(m):
        for j in range(n):
            d = np.abs(xq[i, j] + 1j*yq[i, j] - xy)
            g = np.square(d) * (np.log(d)-1)
            # Value of Green's function at zero
            g[np.where(np.isclose(d, 0))] = 0
            vq[i, j] = (np.expand_dims(g, axis=0) @
                        np.expand_dims(weights, axis=1))[0][0]
    return xq, yq, vq
