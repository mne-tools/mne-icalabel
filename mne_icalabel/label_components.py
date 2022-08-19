from typing import Union

import numpy as np
from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA
from mne.utils import _validate_type
from mne.utils.check import _check_option

from .config import ICALABEL_METHODS
from .iclabel.config import ICLABEL_NUMERICAL_TO_STRING
from .utils._checks import _validate_inst_and_ica


def label_components(inst: Union[BaseRaw, BaseEpochs], ica: ICA, method: str):
    """Automatically label the ICA components with the selected method.

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
    component_dict : dict
        A dictionary with the following fields:

        - 'y_pred_proba' : array of shape (n_components,)
              Estimated predicted probability of the output class
              for each independent component.
        - 'labels': list of shape (n_components,)
              The corresponding string label of each class in 'y_pred'.

    Notes
    -----
    Please refer to the following function for additional information on each
    method:

    - ``'iclabel'``: `~mne_icalabel.iclabel.iclabel_label_components`
    """
    _validate_type(method, str, "method")
    _check_option("method", method, [elt for elt in ICALABEL_METHODS if elt != "manual"])

    _validate_inst_and_ica(inst, ica)
    labels_pred_proba = ICALABEL_METHODS[method](inst, ica)  # type: ignore
    labels_pred = np.argmax(labels_pred_proba, axis=1)
    labels = [ICLABEL_NUMERICAL_TO_STRING[label] for label in labels_pred]
    assert ica.n_components_ == labels_pred.size  # sanity-check
    assert ica.n_components_ == labels_pred_proba.shape[0]  # sanity-check
    y_pred_proba = labels_pred_proba[np.arange(ica.n_components_), labels_pred]

    component_dict = {
        "y_pred_proba": y_pred_proba,
        "labels": labels,
    }
    return component_dict
