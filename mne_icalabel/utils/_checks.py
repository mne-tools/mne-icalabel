import sys
from typing import Tuple, Union

from mne import BaseEpochs
from mne.fixes import _compare_version
from mne.io import BaseRaw
from mne.preprocessing import ICA
from mne.utils import _validate_type, warn


def _validate_inst_and_ica(inst: Union[BaseRaw, BaseEpochs], ica: ICA):
    """Make sure that the provided instance and ICA are valid."""
    _validate_type(inst, (BaseRaw, BaseEpochs), "inst", "Raw or Epochs")
    _validate_type(ica, ICA, "ica")

    if ica.current_fit == "unfitted":
        raise RuntimeError(
            "The provided ICA instance was not fitted. Please use the '.fit()' method to "
            "determine the independent components before trying to label them."
        )


def _check_qt_version() -> Union[Tuple[None, None], Tuple[str, str]]:
    """Check if Qt is available.

    Returns
    -------
    api : str | None
        Which API is used. One of 'PyQt5', 'PyQt6', 'PySide2', 'PySide6'.
    version : str | None
        Version of the API.
    """
    try:
        from qtpy import API_NAME as api
        from qtpy import QtCore
    except Exception:
        api = version = None
    else:
        try:  # pyside
            version = QtCore.__version__
        except AttributeError:
            version = QtCore.QT_VERSION_STR
        if sys.platform == "darwin" and api in ("PyQt5", "PySide2"):
            if not _compare_version(version, ">=", "5.10"):
                warn(
                    f"macOS users should use {api} >= 5.10 for GUIs, "
                    f"got {version}. Please upgrade e.g. with:\n\n"
                    f'    pip install "{api}>=5.10"\n'
                )
    return api, version
