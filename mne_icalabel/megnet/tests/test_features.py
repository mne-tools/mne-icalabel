import numpy as np
import pytest
from mne import create_info
from mne.io import RawArray

from mne_icalabel.megnet.features import _check_line_noise


@pytest.fixture
def raw_with_line_noise():
    """Create a Raw instance with line noise."""
    times = np.arange(0, 2, 1 / 1000)
    data1 = np.sin(2 * np.pi * 10 * times) + np.sin(2 * np.pi * 30 * times)
    data2 = np.sin(2 * np.pi * 30 * times) + np.sin(2 * np.pi * 80 * times)
    data = np.vstack([data1, data2])
    info = create_info(ch_names=["10-30", "30-80"], sfreq=1000, ch_types="mag")
    return RawArray(data, info)


def test_check_line_noise(raw_with_line_noise):
    """Check line-noise auto-detection."""
    assert not _check_line_noise(raw_with_line_noise)
    # 50 Hz is absent from both channels
    raw_with_line_noise.info["line_freq"] = 50
    assert not _check_line_noise(raw_with_line_noise)
    # 10 and 80 Hz are present on one channel each, while 30 Hz is present on both
    raw_with_line_noise.info["line_freq"] = 30
    assert _check_line_noise(raw_with_line_noise)
    raw_with_line_noise.info["line_freq"] = 80
    assert _check_line_noise(raw_with_line_noise)
    raw_with_line_noise.info["line_freq"] = 10
    assert _check_line_noise(raw_with_line_noise)
