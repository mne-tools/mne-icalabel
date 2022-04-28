from typing import Union

from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA
from mne.utils.check import _check_option

from .iclabel import label_components as label_components_iclabel


methods = {
    "iclabel": label_components_iclabel,
}


def label_components(inst: Union[BaseRaw, BaseEpochs], ica: ICA, method: str):
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
