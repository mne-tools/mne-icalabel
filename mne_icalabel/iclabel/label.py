from typing import Union

from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA

from .features import get_features
from .network import run_iclabel


def label(inst: Union[BaseRaw, BaseEpochs], ica: ICA):
    """
    Label the provided ICA components with the ICLabel neural network. This
    network uses 3 features:
        - Topographic maps, based on the ICA decomposition.
        - Power Spectral Density (PSD), based on the ICA decomposition and the
          provided instance.
        - Autocorrelation, based on the ICA decomposition and the provided
          instance.

    Parameters
    ----------
    inst : Raw | Epochs
        Instance used to fit the ICA decomposition. The instance should be
        referenced to a common average and bandpass filtered between 1 and
        100 Hz.
    ica : ICA
        ICA decomposition of the provided instance.
    """
    features = get_features(inst, ica)
    labels = run_iclabel(*features)
    return labels
