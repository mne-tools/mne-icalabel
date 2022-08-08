import numpy as np
import pytest
from mne import create_info
from mne.io import RawArray
from mne.preprocessing import ICA
from mne.utils import requires_version

from mne_icalabel.utils._checks import _validate_inst_and_ica, _check_qt_version


def test_validate_inst_and_ica():
    """Test that validate_inst_and_ica correctly raises."""
    # check types
    with pytest.raises(TypeError, match="inst must be an instance of Raw or Epochs"):
        _validate_inst_and_ica(101, ICA())

    raw = RawArray(np.random.randint(1, 10, (3, 1000)), create_info(3, 100, "eeg"))
    with pytest.raises(TypeError, match="ica must be an instance of ICA"):
        _validate_inst_and_ica(raw, 101)

    # test unfitted
    ica = ICA(n_components=3, method="picard")
    with pytest.raises(RuntimeError, match="The provided ICA instance was not fitted."):
        _validate_inst_and_ica(raw, ica)

    # to avoid RuntimeWarning with fitting an unfiltered raw, let's fake the filter
    with raw.info._unlock():
        raw.info["highpass"] = 1.0
    ica.fit(raw)
    # test valid
    _validate_inst_and_ica(raw, ica)


# TODO: When tests of the GUI are improved and tests on all 4 Qt bindings, the
# test below should pass on different CIs.
# One approach is to:
# - test mne-icalabel except the GUI on CI (1) without Qt instaled
# - test the GUI only on CI (2) on all 4 Qt bindings
# CI (1) can be used to test the 'raise_on_error' and the None return.
# CI (2) can be ujsed to test the 4 functions below.
@requires_version("PyQt5", min_version="")
def test_qt_version_PyQt5():
    """Test _check_qt_version with PyQt5."""
    api, version = _check_qt_version()
    assert api == "PyQt5"
    assert version.split(".")[0] == "5"


@requires_version("PyQt6", min_version="")
def test_qt_version_PyQt6():
    """Test _check_qt_version with PyQt6."""
    api, version = _check_qt_version()
    assert api == "PyQt6"
    assert version.split(".")[0] == "6"


@requires_version("PySide2", min_version="")
def test_qt_version_PySide2():
    """Test _check_qt_version with PySide2."""
    api, version = _check_qt_version()
    assert api == "PySide2"
    assert version.split(".")[0] == "5"


@requires_version("PySide6", min_version="")
def test_qt_version_PySide6():
    """Test _check_qt_version with PySide6."""
    api, version = _check_qt_version()
    assert api == "PySide6"
    assert version.split(".")[0] == "6"
