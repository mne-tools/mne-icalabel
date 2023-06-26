try:
    from importlib.resources import files  # type: ignore
except ImportError:
    from importlib_resources import files  # type: ignore

import numpy as np
import onnxruntime as ort
from numpy.typing import ArrayLike, NDArray

from .utils import _format_input


def _format_input_for_onnx(topo: ArrayLike, psd: ArrayLike, autocorr: ArrayLike):
    """Format the features to the correct shape and type for ONNX."""
    topo = np.transpose(topo, (3, 2, 0, 1)).astype(np.float32)
    psd = np.transpose(psd, (3, 2, 0, 1)).astype(np.float32)
    autocorr = np.transpose(autocorr, (3, 2, 0, 1)).astype(np.float32)

    return topo, psd, autocorr


def _run_iclabel(images: ArrayLike, psds: ArrayLike, autocorr: ArrayLike) -> NDArray:
    """Run ICLabel using onnx."""
    # load weights
    network_file = files("mne_icalabel.iclabel.network") / "assets" / "ICLabelNet.onnx"
    session = ort.InferenceSession(network_file)
    # format inputs
    topo, psds, autocorr = _format_input_for_onnx(
        *_format_input(images, psds, autocorr)
    )
    # run the forward pass
    labels = session.run(
        None,
        {"topo": topo, "psds": psds, "autocorr": autocorr},
    )
    return labels[0]
