# Authors: Adam Li
# License: BSD (3-clause)
#
# This downloads ICLabel model testing data.

import os
import shutil
from functools import partial

import pooch
from mne.datasets import fetch_dataset
from mne.datasets.utils import _mne_path, has_dataset
from mne.utils import verbose

has_icalabel_testing_data = partial(has_dataset, name="icalabel-testing")


@verbose
def data_path(
    path=None, force_update=False, update_path=True, download=True, verbose=None
):  # noqa: D103
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
    update_path : bool | None
        If True, set the MNE_DATASETS_FNIRSMOTORGROUP_PATH in
        mne-python config to the given path. If None, the user is prompted.
    download : bool
        If False and the dataset has not been downloaded yet,
        it will not be downloaded and the path will be returned
        as ‘’ (empty string). This is mostly used for debugging purposes
        and can be safely ignored by most users.
    %(verbose)s

    Returns
    -------
    path : str
        Path to dataset directory.
    """

    dataset_params = dict(
        archive_name="MNE-testing-icalabel-data.zip",
        hash="md5:1ed3a8e12140ef513db8822b6b0dee09",
        url="https://github.com/adam2392/mne-testing-icalabel-data/archive/main.zip",
        folder_name="MNE-testing-icalabel-data",
        dataset_name="icalabel-testing",
        config_key="MNE_DATASETS_ICALABEL_TESTING_PATH",
    )
    folder_name = dataset_params["folder_name"]

    dpath = fetch_dataset(
        dataset_params,
        path=path,
        force_update=force_update,
        update_path=update_path,
        download=download,
        processor=pooch.Unzip(extract_dir=f"./{folder_name}"),
    )
    dpath = str(dpath)

    # Do some wrangling to deal with nested directories
    bad_name = os.path.join(dpath, "mne-testing-icalabel-data-main")
    if os.path.isdir(bad_name):
        os.rename(bad_name, dpath + ".true")
        shutil.rmtree(dpath)
        os.rename(dpath + ".true", dpath)

    return _mne_path(dpath)
