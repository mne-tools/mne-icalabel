from pathlib import Path
from typing import Union

from mne.preprocessing import ICA
from mne.utils import _check_pandas_installed

from ..config import ICLABEL_LABELS_TO_MNE
from ..iclabel.config import ICLABEL_STRING_TO_NUMERICAL


def write_components_tsv(ica: ICA, fname):
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
    `BIDS specification <https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/03-electroencephalography.html#channels-description-_channelstsv>`_  # noqa
    for EEG channels metadata.

    Storage of ICA annotations as a ``.tsv`` file is currently experimental in the
    context of BIDS-EEG Derivatives. The API and functionality is subject to change
    as the community converges on the specification of BIDS-Derivatives.
    """
    from mne_bids import BIDSPath, get_bids_path_from_fname, update_sidecar_json
    from mne_bids.write import _write_json

    pd = _check_pandas_installed(strict=True)

    # ensure the filename is a Path object
    if not isinstance(fname, BIDSPath):
        fname = get_bids_path_from_fname(fname)

    # initialize status, description and IC type
    status = ["good"] * ica.n_components_
    status_description = ["n/a"] * ica.n_components_
    ic_type = ["n/a"] * ica.n_components_

    # extract the component labels if they are present in the ICA instance
    if ica.labels_:
        for label, comps in ica.labels_.items():
            this_status = "good" if label == "brain" else "bad"
            if label in ICLABEL_LABELS_TO_MNE.values():
                for comp in comps:
                    status[comp] = this_status
                    ic_type[comp] = label

    # Create TSV.
    tsv_data = pd.DataFrame(
        dict(
            component=list(range(ica.n_components_)),
            type=["ica"] * ica.n_components_,
            description=["Independent Component"] * ica.n_components_,
            status=status,
            status_description=status_description,
            annotate_method=["n/a"] * ica.n_components_,
            annotate_author=["n/a"] * ica.n_components_,
            ic_type=ic_type,
        )
    )
    # make sure parent directories exist
    fname.mkdir(exist_ok=True)
    tsv_data.to_csv(fname, sep="\t", index=False, encoding="utf-8")

    # create an accompanying JSON file describing the corresponding
    # extra columns for ICA labeling
    component_json = {
        "annotate_method": "Method used for annotating components (e.g. manual, iclabel)",
        "annotate_author": "The name of the person who ran the annotation",
        "ic_type": "The type of annotation must be one of ['brain', "
        "'muscle artifact', 'eye blink', 'heart beat', 'line noise', "
        "'channel noise', 'other']",
    }
    fname = fname.copy().update(extension=".json")

    if not fname.fpath.exists():
        _write_json(fname, component_json)
    else:
        update_sidecar_json(fname, component_json)


def mark_component(
    component: int,
    fname: Union[str, Path],
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
    from mne_bids import BIDSPath, get_bids_path_from_fname

    pd = _check_pandas_installed(strict=True)

    if not isinstance(fname, BIDSPath):
        fname = get_bids_path_from_fname(fname)

    # read the file
    with open(fname, "r") as fin:
        tsv_data = pd.read_csv(fin, sep="\t", index_col=None)

    if component not in tsv_data["component"]:
        raise ValueError(f"Component {component} not in tsv data of {fname}.")

    # check label is correct
    if label not in ICLABEL_STRING_TO_NUMERICAL.keys() and strict_label:
        raise ValueError(
            f"IC annotated label {label} is not a valid label. "
            f"Please use one of {list(ICLABEL_STRING_TO_NUMERICAL.keys())}."
        )
    if label == "brain":
        status = "good"
    else:
        status = "bad"

    # check that method is one of the allowed values
    description = ""
    if method == "manual":
        description += "Manually-annotated "
    else:
        description += f"Auto-detected with {method} "
    description += f"{label}"

    # Read sidecar and create required columns if they do not exist.
    if "status" not in tsv_data:
        tsv_data["status"] = ["good"] * len(tsv_data["component"])
    if "status_description" not in tsv_data:
        tsv_data["status_description"] = ["n/a"] * len(tsv_data["component"])
    for col in ["annotate_method", "annotate_author", "ic_type"]:
        if col not in tsv_data.columns:
            tsv_data[col] = "n/a"

    # load in the tsv file and modify specific columns
    tsv_data.loc[tsv_data["component"] == component, "status"] = status
    tsv_data.loc[tsv_data["component"] == component, "status_description"] = description
    tsv_data.loc[tsv_data["component"] == component, "annotate_method"] = method
    tsv_data.loc[tsv_data["component"] == component, "annotate_author"] = author
    tsv_data.loc[tsv_data["component"] == component, "ic_type"] = label
    tsv_data.to_csv(fname, sep="\t")
