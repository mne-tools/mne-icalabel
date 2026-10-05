import pytest
from mne import create_info
from mne.io import RawArray
from mne.preprocessing import ICA

from mne_icalabel.utils._checks import _validate_ica, _validate_inst_and_ica


def test_validate_inst_and_ica(rng):
    """Test that validate_inst_and_ica correctly raises."""
    # check types
    with pytest.raises(TypeError, match="inst must be an instance of Raw or Epochs"):
        _validate_inst_and_ica(101, ICA())

    raw = RawArray(rng.integers(1, 10, (3, 1000)), create_info(3, 100, "eeg"))
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


def test_validate_ica(rng):
    """Test that _validate_ica correctly raises."""
    # check types
    with pytest.raises(TypeError, match="ica must be an instance of ICA"):
        _validate_ica(101)

    # test unfitted
    ica = ICA(n_components=3, method="picard")
    with pytest.raises(RuntimeError, match="The provided ICA instance was not fitted."):
        _validate_ica(ica)

    # to avoid RuntimeWarning with fitting an unfiltered raw, let's fake the filter
    raw = RawArray(rng.integers(1, 10, (3, 1000)), create_info(3, 100, "eeg"))
    with raw.info._unlock():
        raw.info["highpass"] = 1.0
    ica.fit(raw)
    # test valid
    _validate_ica(ica)
