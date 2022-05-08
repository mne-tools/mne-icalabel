from typing import Union, Optional

import numpy as np
from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA
from mne.utils import _validate_type
from mne.utils.check import _check_option

from .iclabel import iclabel_label_components
from .iclabel.config import ICLABEL_NUMERICAL_TO_STRING
from .utils import _validate_inst_and_ica

methods = {
    "iclabel": iclabel_label_components,
}


def label_components(inst: Union[BaseRaw, BaseEpochs], ica: Optional[ICA] = None, method: str = "auto"):
    """Automatically label the ICA components with the selected method.

    Parameters
    ----------
    inst : Raw | Epochs
        The data instance used to fit the ICA instance.
    ica : ICA | None
        The fitted ICA instance. If None, an ICA decomposition is fitted on the
        provided instance, using the Preconditioned ICA for Real Data (PICARD)
        method and the instance good channels.
    method : str
        The proposed method for labeling components. Must be one of:
        ``"iclabel"``. The default "auto" will use:
            - ``"iclabel"`` for EEG data.

    Returns
    -------
    component_dict : dict
        A dictionary with the following output:
        - 'y_pred_proba' : np.ndarray of shape (n_components,)
        Estimated corresponding predicted probability of the output class
        for each independent component.
        - 'labels': list of shape (n_components,)
        The corresponding string label of each class in 'y_pred'.

    Notes
    -----
    For ICLabel model, the output classes are ordered:
    - 'brain'
    - 'muscle artifact'
    - 'eye blink'
    - 'heart beat'
    - 'line noise'
    - 'channel noise'
    - 'other'
    """
    _validate_type(method, str, "method")
    if method != "auto":
        _check_option("method", method, methods)
    else:
        # TODO: As additional data types are supported, let's define one
        # default method for each datatype.
        #   EEG -> ICLabel
        method = "iclabel"
    _validate_inst_and_ica(inst, ica)
    labels_pred_proba = methods[method](inst, ica)
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
