from typing import Union

from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA
from mne.utils.check import _check_option

from .iclabel import label as iclabel_label


methods = {
    "iclabel": iclabel_label,
}


def label(inst: Union[BaseRaw, BaseEpochs], ica: ICA, method: str):
    """
    Automatically label the ICA components with the selected method.

    Parameters
    ----------
    inst : Raw | Epochs
    ica : ICA
    method : str
    """
    _check_option("method", method, methods)
    return methods[method](inst, ica)
