from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA
from numpy.typing import NDArray

from ..utils._checks import _validate_inst_and_ica as _validate_inst_and_ica
from ._utils import _gdatav4 as _gdatav4
from ._utils import _mne_to_eeglab_locs as _mne_to_eeglab_locs
from ._utils import _next_power_of_2 as _next_power_of_2
from ._utils import _pol2cart as _pol2cart

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
        will be used. See :footcite:t:`iclabel2019` for details.

    References
    ----------
    .. footbibliography::
    """

def _retrieve_eeglab_icawinv(ica: ICA) -> tuple[NDArray[float], NDArray[float]]:
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

def _compute_ica_activations(inst: BaseRaw | BaseEpochs, ica: ICA) -> NDArray[float]:
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
    inst: BaseRaw | BaseEpochs, icawinv: NDArray[float], picks: list[str]
) -> NDArray[float]:
    """Topoplot feature."""

def _topoplotFast(
    values: NDArray[float], rd: NDArray[float], th: NDArray[float]
) -> NDArray[float]:
    """Implement topoplotFast.m from MATLAB. Each topographic map is a 32x32 images."""

def _eeg_rpsd(
    inst: BaseRaw | BaseEpochs, ica: ICA, icaact: NDArray[float]
) -> NDArray[float]:
    """PSD feature."""

def _eeg_rpsd_constants(
    inst: BaseRaw | BaseEpochs, ica: ICA
) -> tuple[int, int, int, int, NDArray[int], NDArray[float], NDArray[int]]:
    """Compute the constants before ``randperm`` is used to compute the subset."""

def _eeg_rpsd_compute_psdmed(
    inst: BaseRaw | BaseEpochs,
    icaact: NDArray[float],
    ncomp: int,
    nfreqs: int,
    n_points: int,
    nyquist: int,
    index: NDArray[int],
    window: NDArray[float],
    subset: NDArray[int],
) -> NDArray[float]:
    """Compute the variable 'psdmed', annotated as windowed spectrums."""

def _eeg_rpsd_format(psd: NDArray[float]) -> NDArray[float]:
    """Apply the formatting steps after 'eeg_rpsd.m'."""

def _eeg_autocorr_welch(
    raw: BaseRaw, ica: ICA, icaact: NDArray[float]
) -> NDArray[float]:
    """Autocorrelation feature applied on raw object with at least 5 * fs samples.

    MATLAB: 'eeg_autocorr_welch.m'.
    """

def _eeg_autocorr(raw: BaseRaw, ica: ICA, icaact: NDArray[float]) -> NDArray[float]:
    """Autocorr applied on raw object without enough sampes for eeg_autocorr_welch.

    MATLAB: 'eeg_autocorr.m'.
    """

def _eeg_autocorr_fftw(
    epochs: BaseEpochs, ica: ICA, icaact: NDArray[float]
) -> NDArray[float]:
    """Autocorrelation feature applied on epoch object.

    MATLAB: 'eeg_autocorr_fftw.m'.
    """

def _resample(ac: NDArray[float], fs: int | float) -> NDArray[float]:
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
