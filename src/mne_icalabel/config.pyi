from collections.abc import Callable as Callable

from .iclabel import iclabel_label_components as iclabel_label_components
from .iclabel._config import ICLABEL_NUMERICAL_TO_STRING as ICLABEL_NUMERICAL_TO_STRING
from .iclabel._config import ICLABEL_STRING_TO_NUMERICAL as ICLABEL_STRING_TO_NUMERICAL
from .megnet import megnet_label_components as megnet_label_components
from .megnet._config import MEGNET_NUMERICAL_TO_STRING as MEGNET_NUMERICAL_TO_STRING
from .megnet._config import MEGNET_STRING_TO_NUMERICAL as MEGNET_STRING_TO_NUMERICAL

ICALABEL_METHODS: dict[str, Callable | None]
ICALABEL_METHODS_NUMERICAL_TO_STRING: dict[str, dict[int, str]]
ICALABEL_METHODS_STRING_TO_NUMERICAL: dict[str, dict[str, int]]
ICA_LABELS_TO_MNE: dict[str, str]
