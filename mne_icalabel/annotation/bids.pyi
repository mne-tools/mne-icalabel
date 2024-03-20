from pathlib import Path as Path

from mne.preprocessing import ICA

from ..config import ICA_LABELS_TO_MNE as ICA_LABELS_TO_MNE
from ..iclabel._config import ICLABEL_STRING_TO_NUMERICAL as ICLABEL_STRING_TO_NUMERICAL
from ..utils._imports import import_optional_dependency as import_optional_dependency

def write_components_tsv(ica: ICA, fname: str | Path):
    """Write channels tsv file for ICA components.

    Will create an accompanying JSON sidecar to explain the
    additional columns created in the channels tsv file for the
    ICA component labeling.

    Parameters
    ----------
    ica : ICA
        An instance of the fitted ICA.
    fname : str | Path
        The output filename.

    Notes
    -----
    Components are stored in a ``.tsv`` file essentially in the same manner as
    ``channels.tsv`` files for BIDS-EEG data. For more information, see the
    `BIDS specification <https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/03-electroencephalography.html#channels-description-_channelstsv>`_
    for EEG channels metadata.

    Storage of ICA annotations as a ``.tsv`` file is currently experimental in the
    context of BIDS-EEG Derivatives. The API and functionality is subject to change
    as the community converges on the specification of BIDS-Derivatives.
    """

def mark_component(
    component: int,
    fname: str | Path,
    method: str,
    label: str,
    author: str,
    strict_label: bool = True,
):
    """Mark a component with a label.

    Parameters
    ----------
    component : int
        The component to mark.
    fname : Union[str, Path]
        The filename for the BIDS filepath.
    method : str
        The method to use. Must be 'manual', or one of ['iclabel'].
    label : str
        The label of the ICA component. Must be one of ['brain',
        'muscle artifact', 'eye blink', 'heart beat', 'line noise',
        'channel noise', 'other'].
    author : str
        The annotating author.
    strict_label : bool
        Whether to raise an error if ``label`` is not an accepted value.
        Default is True.

    Notes
    -----
    Storage of ICA annotations as a ``.tsv`` file is currently experimental in the
    context of BIDS-EEG Derivatives. The API and functionality is subject to change
    as the community converges on the specification of BIDS-Derivatives.
    """
