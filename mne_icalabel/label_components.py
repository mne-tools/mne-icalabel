from typing import Union

from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA
from mne.utils import _validate_type
from mne.utils.check import _check_option

from .iclabel import label_components as label_components_iclabel
from .utils import _validate_inst_and_ica

methods = {
    "iclabel": label_components_iclabel,
}


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
        The proposed method for labeling components. Must be one of
        ('iclabel', ).

    Returns
    -------
    labels : np.ndarray of shape (n_components,) or (n_components, n_class)
        The estimated numerical labels of each ICA component.
    """
    _validate_type(method, str, 'method')
    _check_option("method", method, methods)
    _validate_inst_and_ica(inst, ica)
    return methods[method](inst, ica)
