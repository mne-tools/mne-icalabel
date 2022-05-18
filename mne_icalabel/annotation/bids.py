from pathlib import Path
from typing import Union

from mne.preprocessing import ICA
from mne.utils import _check_pandas_installed

from ..iclabel.config import ICLABEL_STRING_TO_NUMERICAL


def write_channels_tsv(ica: ICA, fname: Union[str, Path]):
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
    """
    from mne_bids import get_bids_path_from_fname, update_sidecar_json
    from mne_bids.write import _write_json

    pd = _check_pandas_installed(strict=True)

    # ensure the filename is a Path object
    fname = Path(fname)
    fname = get_bids_path_from_fname(fname)

    # Create TSV.
    tsv_data = pd.DataFrame(
        dict(
            component=list(range(ica.n_components_)),
            type=["ica"] * ica.n_components_,
            description=["Independent Component"] * ica.n_components_,
            status=["good"] * ica.n_components_,
            status_description=["n/a"] * ica.n_components_,
            annotate_method=["n/a"] * ica.n_components_,
            annotate_author=["n/a"] * ica.n_components_,
            ic_type=["n/a"] * ica.n_components_,
        )
    )

    tsv_data.to_csv(fname, sep="\t", index=False)

    # create an accompanying JSON file describing the corresponding
    # extra columns for ICA labeling
    component_json = {
        "annotate_method": "Method used for annotating components (e.g. manual, iclabel)",
        "annotate_author": "The name of the person who ran the annotation",
        "ic_type": "The type of annotation must be one of ['brain', "
        "'muscle artifact', 'eye blink', 'heart beat', 'line noise', "
        "'channel noise', 'other']",
    }
    fname = fname.update(extension=".json")
    if not fname.exists():
        _write_json(fname, component_json)
    else:
        update_sidecar_json(fname, component_json)


def mark_component(component: int, fname: Union[str, Path], method: str, label: str, author: str):
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

    Raises
    ------
    ValueError
        _description_
    """
    from mne_bids import get_bids_path_from_fname, mark_channels

    pd = _check_pandas_installed(strict=True)

    bids_path = get_bids_path_from_fname(fname)

    if label not in ICLABEL_STRING_TO_NUMERICAL.keys():
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

    mark_channels(
        bids_path=bids_path, ch_names=str(component), status=status, descriptions=description
    )

    # load in the tsv file and modify specific columns
    tsv_data = pd.read_csv(bids_path, sep="\t")
    tsv_data[tsv_data["component"] == component]["annotate_method"] = method
    tsv_data[tsv_data["component"] == component]["annotate_author"] = author
    tsv_data[tsv_data["component"] == component]["ic_type"] = label
