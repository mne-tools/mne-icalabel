from typing import Union

import numpy as np
from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA

from .features import get_iclabel_features
from .network import run_iclabel


def iclabel_label_components(inst: Union[BaseRaw, BaseEpochs], ica: ICA, inplace: bool = True):
    """Label the provided ICA components with the ICLabel neural network.

    ICLabel is designed to classify ICs fitted with an extended infomax ICA
    decomposition algorithm on EEG datasets referenced to a common average and
    filtered between [1., 100.] Hz. It is possible to run ICLabel on datasets that
    do not meet those specification, but the classification performance
    might be negatively impacted. Moreover, the ICLabel paper did not study the
    effects of these preprocessing steps.

    ICLabel uses 3 features:

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
        ICA decomposition of the provided instance. The ICA decomposition
        should use the extended infomax method.
    inplace : bool
        Whether to modify the ``ica`` instance in place by adding the automatic
        annotations to the ``labels_`` property. By default True.

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

    if inplace:
        from mne_icalabel.config import ICLABEL_LABELS_TO_MNE

        ica.labels_scores_ = labels_pred_proba
        argmax_labels = np.argmax(labels_pred_proba, axis=1)

        # add labels to the ICA instance
        for idx, (_, mne_label) in enumerate(ICLABEL_LABELS_TO_MNE.items()):
            auto_labels = list(np.argwhere(argmax_labels == idx).flatten())
            if mne_label not in ica.labels_:
                ica.labels_[mne_label] = auto_labels
                continue
            for comp in auto_labels:
                if comp not in ica.labels_[mne_label]:
                    ica.labels_[mne_label].append(comp)

    return labels_pred_proba
