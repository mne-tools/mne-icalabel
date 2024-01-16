import torch
import torch.nn as nn
from _typeshed import Incomplete
from numpy.typing import ArrayLike, NDArray

from .utils import _format_input as _format_input

class _ICLabelNetImg(nn.Module):
    conv1: Incomplete
    relu1: Incomplete
    conv2: Incomplete
    relu2: Incomplete
    conv3: Incomplete
    relu3: Incomplete
    sequential: Incomplete

    def __init__(self) -> None:
        ...

    def forward(self, x):
        ...

class _ICLabelNetPSDS(nn.Module):
    conv1: Incomplete
    relu1: Incomplete
    conv2: Incomplete
    relu2: Incomplete
    conv3: Incomplete
    relu3: Incomplete
    sequential: Incomplete

    def __init__(self) -> None:
        ...

    def forward(self, x):
        ...

class _ICLabelNetAutocorr(nn.Module):
    conv1: Incomplete
    relu1: Incomplete
    conv2: Incomplete
    relu2: Incomplete
    conv3: Incomplete
    relu3: Incomplete
    sequential: Incomplete

    def __init__(self) -> None:
        ...

    def forward(self, x):
        ...

class ICLabelNet(nn.Module):
    """The ICLabel neural network."""
    img_conv: Incomplete
    psds_conv: Incomplete
    autocorr_conv: Incomplete
    conv: Incomplete
    softmax: Incomplete
    seq: Incomplete

    def __init__(self) -> None:
        ...

    @staticmethod
    def reshape_fortran(x: torch.Tensor, shape) -> torch.Tensor:
        ...

    def reshape_concat(self, tensor: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, images: torch.Tensor, psds: torch.Tensor, autocorr: torch.Tensor) -> torch.Tensor:
        ...

def _format_input_for_torch(topo: ArrayLike, psd: ArrayLike, autocorr: ArrayLike):
    """Format the features to the correct shape and type for pytorch."""

def _run_iclabel(images: ArrayLike, psds: ArrayLike, autocorr: ArrayLike) -> NDArray:
    """Run ICLabel using onnx."""