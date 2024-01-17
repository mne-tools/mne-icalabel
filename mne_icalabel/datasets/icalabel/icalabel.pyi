from pathlib import Path
from typing import Optional

from _typeshed import Incomplete

has_icalabel_testing_data: Incomplete

def data_path(
    path: Optional[str] = None,
    force_update: bool = False,
    update_path: bool = True,
    download: bool = True,
    verbose: Incomplete | None = None,
) -> Path:
    """ICA label testing data generated in conjunction with EEGLab.

    Parameters
    ----------
    path : None | str
        Location of where to look for the dataset.
        If None, the environment variable or config parameter is used.
        If it doesn’t exist, the “~/mne_data” directory is used.
        If the dataset is not found under the given path,
        the data will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool
        If True, set the MNE_DATASETS_FNIRSMOTORGROUP_PATH in
        mne-python config to the given path.
    download : bool
        If False and the dataset has not been downloaded yet,
        it will not be downloaded and the path will be returned
        as ‘’ (empty string). This is mostly used for debugging purposes
        and can be safely ignored by most users.

    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    path : Path
        Path to dataset directory.
    """
