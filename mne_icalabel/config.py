from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING

from .iclabel import iclabel_label_components

if TYPE_CHECKING:
    from typing import Callable, Optional

ICALABEL_METHODS: dict[str, Optional[Callable]] = {
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
