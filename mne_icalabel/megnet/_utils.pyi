from numpy.typing import NDArray

def _make_head_outlines(sphere: NDArray, pos: NDArray, clip_origin: tuple) -> dict:
    """Generate head outlines for topomap plotting.

    This is a modified version of mne.viz.topomap._make_head_outlines.
    The difference between this function and the original one is that
    head_x and head_y here are scaled by a factor of 1.01 to make topomap
    fit the 120x120 pixel size.
    Also, removed the ear and nose outlines, not needed in MEGnet.

    Parameters
    ----------
    sphere : NDArray
    The sphere parameters (x, y, z, radius).
    pos : NDArray
    The 2D sensor positions.
    clip_origin : tuple
    The origin of the clipping circle.

    Returns
    -------
    dict
    Dictionary containing the head outlines and mask positions.
    """
