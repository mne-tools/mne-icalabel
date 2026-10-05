from numpy.typing import ArrayLike, NDArray

def run_iclabel(
    images: ArrayLike,
    psds: ArrayLike,
    autocorr: ArrayLike,
    backend: str | None = "pytorch",
) -> NDArray:
    """Run the ICLabel network on the provided set of features.

    The features are un-formatted and are as-returned by
    `~mne_icalabel.iclabel.get_iclabel_features`. For more information,
    see :footcite:t:`PionTonachini2019`.

    Parameters
    ----------
    images : array of shape (n_components, 1, 32, 32)
    The topoplot images.
    psds : array of shape (n_components, 1, 1, 100)
    The power spectral density features.
    autocorr : array of shape (n_components, 1, 1, 100)
    The autocorrelation features.
    backend : None | ``torch`` | ``onnx``
    Backend to use to run ICLabel. If None, returns the first available backend in
    the order ``torch``, ``onnx``.

    Returns
    -------
    labels : array of shape (n_components, n_classes)
    The predicted numerical probability values for all labels in ICLabel output.
    Columns are ordered with ``'Brain'``, ``'Muscle'``, ``'Eye'``,
    ``'Heart'``, ``'Line Noise'``, ``'Channel Noise'``, and ``'Other'``.

    References
    ----------
    .. footbibliography::
    """
