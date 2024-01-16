from typing import Callable, Optional

from .iclabel import iclabel_label_components as iclabel_label_components

ICALABEL_METHODS: dict[str, Optional[Callable]]
ICA_LABELS_TO_MNE: dict[str, str]