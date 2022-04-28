"""ICLabel - An automated electroencephalographic independent component
classifier, dataset, and website.

This is a python implementation of the EEGLAB plugin 'ICLabel'."""

from .features import get_features  # noqa: F401
from .label_components import label_components  # noqa: F401
from .network import ICLabelNet, run_iclabel  # noqa: F401
