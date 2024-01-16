from numpy.typing import ArrayLike, NDArray

from .utils import _format_input as _format_input

def _format_input_for_onnx(topo: ArrayLike, psd: ArrayLike, autocorr: ArrayLike):
    """Format the features to the correct shape and type for ONNX."""

def _run_iclabel(images: ArrayLike, psds: ArrayLike, autocorr: ArrayLike) -> NDArray:
    """Run ICLabel using onnx."""