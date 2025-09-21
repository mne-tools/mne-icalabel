from __future__ import annotations

from typing import TYPE_CHECKING

from .iclabel import iclabel_label_components

if TYPE_CHECKING:
    from collections.abc import Callable

ICALABEL_METHODS: dict[str, Callable | None] = {
    "iclabel": iclabel_label_components,
    "manual": None,
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
