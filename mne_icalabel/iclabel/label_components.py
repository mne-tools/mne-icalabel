from typing import Union

from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA

from .features import get_iclabel_features
from .network import run_iclabel


def iclabel_label_components(inst: Union[BaseRaw, BaseEpochs], ica: ICA):
    """Label the provided ICA components with the ICLabel neural network.

    This network uses 3 features:

    - Topographic maps, based on the ICA decomposition.
    - Power Spectral Density (PSD), based on the ICA decomposition and the
      provided instance.
    - Autocorrelation, based on the ICA decomposition and the provided
      instance.

    For more information, see :footcite:`iclabel2019`.

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
    labels_pred_proba : numpy.ndarray of shape (n_components, n_classes)
        The estimated corresponding predicted probabilities of output classes
        for each independent component. Columns are ordered with 'brain',
        'muscle artifact', 'eye blink', 'heart beat', 'line noise',
        'channel noise', 'other'.

    References
    ----------
    .. footbibliography::
    """
    features = get_iclabel_features(inst, ica)
    labels_pred_proba = run_iclabel(*features)
    return labels_pred_proba
