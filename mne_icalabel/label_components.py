from typing import Union

import numpy as np
from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA
from mne.utils import _validate_type
from mne.utils.check import _check_option
from sklearn.base import TransformerMixin, BaseEstimator

from .iclabel import label_components as label_components_iclabel
from .utils import _validate_inst_and_ica

methods = {
    "iclabel": label_components_iclabel,
}

class AutoLabelICA(TransformerMixin):
    def __init__(self, method:str ='iclabel') -> None:
        self.method = method

    def fit(self, X, y):
        pass

    def transform(self, raw, ica):
        ic_labels = label_components(raw, ica, method=self.method)

        # Afterwards, we can hard threshold the probability values to assign
        # each component to be kept or not (i.e. it is part of brain signal).
        # The first component was visually an artifact, which was captured
        # for certain.
        not_brain_index = np.argmax(ic_labels, axis=1) != 0
        exclude_idx = np.argwhere(not_brain_index).squeeze()

        ica.apply(raw, exclude=exclude_idx)
        return raw


def label_components(inst: Union[BaseRaw, BaseEpochs], ica: ICA, method: str):
    """
    Automatically label the ICA components with the selected method.

    Parameters
    ----------
    inst : Raw | Epochs
        The data instance used to fit the ICA instance.
    ica : ICA
        The fitted ICA instance.
    method : str
        The proposed method for labeling components. Must be one of:
        ``'iclabel'``.

    Returns
    -------
    labels : np.ndarray of shape (n_components,) or (n_components, n_classes)
        The estimated corresponding predicted probabilities of output classes
        for each independent component.

    Notes
    -----
    For ICLabel model, the output classes are ordered:
    - 'Brain'
    - 'Muscle'
    - 'Eye'
    - 'Heart'
    - 'Line Noise'
    - 'Channel Noise'
    - 'Other'
    """
    _validate_type(method, str, "method")
    _check_option("method", method, methods)
    _validate_inst_and_ica(inst, ica)
    return methods[method](inst, ica)
