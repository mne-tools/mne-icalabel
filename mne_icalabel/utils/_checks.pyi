from typing import Union

from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA

def _validate_inst_and_ica(inst: Union[BaseRaw, BaseEpochs], ica: ICA):
    """Make sure that the provided instance and ICA are valid."""

def _validate_ica(ica: ICA):
    """Make sure that the provided ICA is valid."""
