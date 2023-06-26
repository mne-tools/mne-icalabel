from numpy.typing import ArrayLike, NDArray


def run_iclabel(images: ArrayLike, psds: ArrayLike, autocorr: ArrayLike, library: str = "pytorch") -> NDArray:
    """Run the ICLabel network on the provided set of features.

    The features are un-formatted and are as-returned by
    `~mne_icalabel.iclabel.get_iclabel_features`.

    Parameters
    ----------
    images : array of shape (n_components, 1, 32, 32)
        The topoplot images.
    psds : array of shape (n_components, 1, 1, 100)
        The power spectral density features.
    autocorr : array of shape (n_components, 1, 1, 100)
        The autocorrelation features.

    Returns
    -------
    labels : array of shape (n_components, n_classes)
        The predicted numerical probability values for all labels in ICLabel output.
        Columns are ordered with ``'Brain'``, ``'Muscle'``, ``'Eye'``,
        ``'Heart'``, ``'Line Noise'``, ``'Channel Noise'``, and ``'Other'``.
    """
    if library == "pytorch":
        from .torch import _run_iclabel
    elif library == "onnx":
        from .onnx import _run_iclabel
    else:
        raise ValueError(f"Library {library} is not supported")
    return _run_iclabel(images, psds, autocorr)
