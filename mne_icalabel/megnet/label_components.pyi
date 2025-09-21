import onnxruntime as ort
from mne.io import BaseRaw
from mne.preprocessing import ICA
from numpy.typing import NDArray

from .features import get_megnet_features as get_megnet_features

_MODEL_PATH: str

def megnet_label_components(raw: BaseRaw, ica: ICA) -> NDArray:
    """Label the provided ICA components with the MEGnet neural network.

    For more information, see :footcite:t:`Treacher2021`.

    Parameters
    ----------
    raw : Raw
    Raw MEG recording used to fit the ICA decomposition.
    The raw instance should be bandpass filtered between ``1`` and ``100`` Hz
    and notch filtered at ``50`` or ``60`` Hz to remove line noise, and downsampled
    to ``250`` Hz.
    ica : ICA
    ICA decomposition of the provided instance.
    The ICA decomposition should use the infomax method.

    Returns
    -------
    labels_pred_proba : numpy.ndarray of shape (n_components, n_classes)
    The estimated corresponding predicted probabilities of output classes
    for each independent component. Columns are ordered with
    ``'brain/other'``, ``'eye movement'``, ``'heart beat'``, ``'eye blink'``.

    References
    ----------
    .. footbibliography::
    """

def _chunk_predicting(
    session: ort.InferenceSession,
    time_series: NDArray,
    spatial_maps: NDArray,
    chunk_len: int = 15000,
    overlap_len: int = 3750,
) -> NDArray:
    """MEGnet's chunk volte algorithm."""

def _get_chunk_start(
    input_len: int, chunk_len: int = 15000, overlap_len: int = 3750
) -> list:
    """Calculate start times for time series chunks with overlap."""
