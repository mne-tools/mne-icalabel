from typing import Union

from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA
from mne.utils import _validate_type


def _validate_inst_and_ica(inst: Union[BaseRaw, BaseEpochs], ica: ICA):
    """Make sure that the provided instance and ICA are valid."""
    _validate_type(inst, (BaseRaw, BaseEpochs), "inst", "Raw or Epochs")
    _validate_type(ica, ICA, "ica")

    if ica.current_fit == "unfitted":
        raise RuntimeError(
            "The provided ICA instance was not fitted. Please use the '.fit()' method to "
            "determine the independent components before trying to label them."
        )
