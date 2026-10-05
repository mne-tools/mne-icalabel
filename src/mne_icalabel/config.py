from __future__ import annotations

from typing import TYPE_CHECKING

from .iclabel import iclabel_label_components
from .iclabel._config import ICLABEL_NUMERICAL_TO_STRING, ICLABEL_STRING_TO_NUMERICAL
from .megnet import megnet_label_components
from .megnet._config import MEGNET_NUMERICAL_TO_STRING, MEGNET_STRING_TO_NUMERICAL

if TYPE_CHECKING:
    from collections.abc import Callable

ICALABEL_METHODS: dict[str, Callable | None] = {
    "iclabel": iclabel_label_components,
    "megnet": megnet_label_components,
    "manual": None,
}

ICALABEL_METHODS_NUMERICAL_TO_STRING: dict[str, dict[int, str]] = {
    "iclabel": ICLABEL_NUMERICAL_TO_STRING,
    "megnet": MEGNET_NUMERICAL_TO_STRING,
}
ICALABEL_METHODS_STRING_TO_NUMERICAL: dict[str, dict[str, int]] = {
    "iclabel": ICLABEL_STRING_TO_NUMERICAL,
    "megnet": MEGNET_STRING_TO_NUMERICAL,
}

# map labels to the equivalent str format in MNE
ICA_LABELS_TO_MNE: dict[str, str] = {
    "Brain": "brain",
    "Muscle": "muscle",
    "Eye": "eog",
    "Heart": "ecg",
    "Line Noise": "line_noise",
    "Channel Noise": "ch_noise",
    "Other": "other",
}
