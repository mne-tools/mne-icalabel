from typing import Union

from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA

from ..utils import _validate_inst_and_ica
from .features import get_iclabel_features
from .network import run_iclabel


def label_components(inst: Union[BaseRaw, BaseEpochs], ica: ICA):
    """Label the provided ICA components with the ICLabel neural network.

    This network uses 3 features:

    - Topographic maps, based on the ICA decomposition.
    - Power Spectral Density (PSD), based on the ICA decomposition and the
      provided instance.
    - Autocorrelation, based on the ICA decomposition and the provided
      instance.

    For more information, see :footcite:`iclabel2019`

    Parameters
    ----------
    inst : Raw | Epochs
        Instance used to fit the ICA decomposition. The instance should be
        referenced to a common average and bandpass filtered between 1 and
        100 Hz.
    ica : ICA
        ICA decomposition of the provided instance.

    Returns
    -------
    labels : numpy.ndarray of shape (n_components,)
        The estimated corresponding numerical labels for each independent
        component.

    References
    ----------
    .. footbibliography::
    """
    _validate_inst_and_ica(inst, ica)
    features = get_iclabel_features(inst, ica)
    labels = run_iclabel(*features)
    return labels
