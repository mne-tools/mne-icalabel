from mne.io import BaseRaw
from mne.preprocessing import ICA
from numpy.typing import NDArray

from ..utils.transform import cart2sph as cart2sph
from ..utils.transform import pol2cart as pol2cart
from ._utils import _make_head_outlines as _make_head_outlines

def get_megnet_features(raw: BaseRaw, ica: ICA):
    """Extract time series and topomaps for each ICA component.

    MEGNet uses topomaps from BrainStorm exported as 120x120x3 RGB images.
    Thus, we need to replicate the 'appearance'/'look' of a BrainStorm topomap.

    Parameters
    ----------
    raw : Raw
    Raw MEG recording used to fit the ICA decomposition.
    The raw instance should be bandpass filtered between
    1 and 100 Hz and notch filtered at 50 or 60 Hz to
    remove line noise, and downsampled to 250 Hz.
    ica : ICA
    ICA decomposition of the provided instance.
    The ICA decomposition should use the infomax method.

    Returns
    -------
    time_series : array of shape (n_components, n_samples)
    The time series for each ICA component.
    topomaps : array of shape (n_components, 120, 120, 3)
    The topomap RGB images for each ICA component.
    """

def _get_topomaps_data(ica: ICA):
    """Prepare 2D sensor positions and outlines for topomap plotting."""

def _get_topomaps(ica: ICA, pos_new: NDArray, outlines: dict):
    """Generate topomap images for each ICA component."""

def _check_line_noise(
    raw: BaseRaw, *, neighbor_width: int = 4, threshold_factor: int = 10
) -> bool:
    """Check if line noise is present in the MEG/EEG data."""
