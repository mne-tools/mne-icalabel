from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA

from .config import ICALABEL_METHODS as ICALABEL_METHODS
from .iclabel._config import ICLABEL_NUMERICAL_TO_STRING as ICLABEL_NUMERICAL_TO_STRING
from .utils._checks import _validate_inst_and_ica as _validate_inst_and_ica

def label_components(inst: BaseRaw | BaseEpochs, ica: ICA, method: str):
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
