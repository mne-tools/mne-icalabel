from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA
from numpy.typing import NDArray

from ..utils._checks import _validate_inst_and_ica as _validate_inst_and_ica
from ..utils.transform import pol2cart as pol2cart
from ._utils import _gdatav4 as _gdatav4
from ._utils import _mne_to_eeglab_locs as _mne_to_eeglab_locs
from ._utils import _next_power_of_2 as _next_power_of_2

def get_iclabel_features(inst: BaseRaw | BaseEpochs, ica: ICA):
    """Generate the features for ICLabel neural network.

    Parameters
    ----------
    inst : Raw | Epochs
    MNE Raw/Epoch instance with data array in Volts.
    ica : ICA
    MNE ICA decomposition.

    Returns
    -------
    topo : array of shape (32, 32, 1, n_components)
    The topoplot feature.
    psd : array of shape (1, 100, 1, n_components)
    The psd feature.
    autocorr : array of shape (1, 100, 1, n_components)
    The autocorrelations feature. Depending on the length of the
    raw data passed in, different methods of computing autocorrelation
    will be used. See :footcite:t:`PionTonachini2019` for details.

    References
    ----------
    .. footbibliography::
    """

def _retrieve_eeglab_icawinv(ica: ICA) -> tuple[NDArray, NDArray]:
    """
    Retrieve 'icawinv' from an MNE ICA instance.

    Parameters
    ----------
    ica : ICA
    MNE ICA decomposition.

    Returns
    -------
    icawinv : array
    weights : array
    """

def _compute_ica_activations(inst: BaseRaw | BaseEpochs, ica: ICA) -> NDArray:
    """Compute the ICA activations 'icaact' variable from an MNE ICA instance.

    Parameters
    ----------
    inst : Raw | Epoch
    MNE Raw/Epoch instance with data array in Volts.
    ica : ICA
    MNE ICA decomposition.

    Returns
    -------
    icaact : array
    raw: (n_components, n_samples)
    epoch: (n_components, n_samples, n_trials)

    Notes
    -----
    In EEGLAB, the ICA activations are computed after the data and the ICA
    decomposition are re-referenced to a common average, if the field 'EEG.ref'
    is different from 'averef'. The EEGLAB sample dataset's field 'EEG.ref' is
    set to 'common', thus triggering the re-referencing with 'pop_reref' which
    seems to be buggy and breaks the ICA solution. After 'pop_reref' is called,
    the relation 'inv(EEG.icaweights * EEG.icasphere) = EEG.icawinv' is not
    respected anymore.

    Additionally, 'pop_reref' changes the field 'EEG.ref' to 'average'. It is
    assumed that 'common', 'average' and 'averef' are all denoting a common
    average reference.
    """

def _eeg_topoplot(
    inst: BaseRaw | BaseEpochs, icawinv: NDArray, picks: list[str]
) -> NDArray:
    """Topoplot feature."""

def _topoplotFast(values: NDArray, rd: NDArray, th: NDArray) -> NDArray:
    """Implement topoplotFast.m from MATLAB. Each topographic map is a 32x32 images."""

def _eeg_rpsd(inst: BaseRaw | BaseEpochs, ica: ICA, icaact: NDArray) -> NDArray:
    """PSD feature."""

def _eeg_rpsd_constants(
    inst: BaseRaw | BaseEpochs, ica: ICA
) -> tuple[int, int, int, int, NDArray, NDArray, NDArray]:
    """Compute the constants before ``randperm`` is used to compute the subset."""

def _eeg_rpsd_compute_psdmed(
    inst: BaseRaw | BaseEpochs,
    icaact: NDArray,
    ncomp: int,
    nfreqs: int,
    n_points: int,
    nyquist: int,
    index: NDArray,
    window: NDArray,
    subset: NDArray,
) -> NDArray:
    """Compute the variable 'psdmed', annotated as windowed spectrums."""

def _eeg_rpsd_format(psd: NDArray) -> NDArray:
    """Apply the formatting steps after 'eeg_rpsd.m'."""

def _eeg_autocorr_welch(raw: BaseRaw, ica: ICA, icaact: NDArray) -> NDArray:
    """Autocorrelation feature applied on raw object with at least 5 * fs samples.

    MATLAB: 'eeg_autocorr_welch.m'.
    """

def _eeg_autocorr(raw: BaseRaw, ica: ICA, icaact: NDArray) -> NDArray:
    """Autocorr applied on raw object without enough samples for eeg_autocorr_welch.

    MATLAB: 'eeg_autocorr.m'.
    """

def _eeg_autocorr_fftw(epochs: BaseEpochs, ica: ICA, icaact: NDArray) -> NDArray:
    """Autocorrelation feature applied on epoch object.

    MATLAB: 'eeg_autocorr_fftw.m'.
    """

def _resample(ac: NDArray, fs: int | float) -> NDArray:
    """Resample the autocorrelation feature.

    The comment in EEGLAB is:
    resample to 1 second at 100 samples/sec

    Which translates by: the output array must be of shape (n_comp, 101), thus
    the resampling up variable is set to 100, and down variable must respect:
    100 < ac.T.shape[0] * 100 / down <= 101
    If the instance sampling frequency is an integer, then down is equal to the
    sampling frequency.

    Parameters
    ----------
    ac : array
    Array of shape (n_comp, samples).
    fs : int | float
    Sampling frequency of the MNE instance.
    """
