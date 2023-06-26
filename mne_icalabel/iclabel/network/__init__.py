from typing import Optional

from mne.utils import _check_option
from numpy.typing import ArrayLike, NDArray

from ...utils._imports import import_optional_dependency


def run_iclabel(images: ArrayLike, psds: ArrayLike, autocorr: ArrayLike, backend: Optional[str] = "pytorch") -> NDArray:
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
    backend : None | ``torch`` | ``onnx``
        Backend to use to run ICLabel. If None, returns the first available backend in
        the order ``torch``, ``onnx``.

    Returns
    -------
    labels : array of shape (n_components, n_classes)
        The predicted numerical probability values for all labels in ICLabel output.
        Columns are ordered with ``'Brain'``, ``'Muscle'``, ``'Eye'``,
        ``'Heart'``, ``'Line Noise'``, ``'Channel Noise'``, and ``'Other'``.
    """
    _check_option(backend, (None, "torch", "onnx"), "backend")
    if backend is None:
        torch = import_optional_dependency("torch", raise_error=False)
        onnx = import_optional_dependency("onnxruntime", raise_error=False)

        if torch is not None:
            from .torch import _run_iclabel
        elif torch is None and onnx is not None:
            from .onnx import _run_iclabel
        else:
            raise ImportError(
                "Missing optional dependency. ICLabel requires either pytorch or "
                "onnxruntime. Use pip or conda to install one of them."
            )
    elif backend == "torch":
        import_optional_dependency("torch", raise_error=True)

        from .torch import _run_iclabel
    elif backend == "onnx":
        import_optional_dependency("onnxruntime", raise_error=True)

        from .onnx import _run_iclabel
    return _run_iclabel(images, psds, autocorr)
