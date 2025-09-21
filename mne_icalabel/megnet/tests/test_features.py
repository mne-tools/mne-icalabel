import numpy as np
import pytest
from mne import create_info
from mne.io import RawArray
from mne.preprocessing import ICA

from mne_icalabel.megnet.features import _check_line_noise, get_megnet_features


@pytest.fixture
def raw_with_line_noise() -> RawArray:
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
    # 10 and 80 Hz are present on one channel each,
    # while 30 Hz is present on both
    raw_with_line_noise.info["line_freq"] = 30
    assert _check_line_noise(raw_with_line_noise)
    raw_with_line_noise.info["line_freq"] = 80
    assert _check_line_noise(raw_with_line_noise)
    raw_with_line_noise.info["line_freq"] = 10
    assert _check_line_noise(raw_with_line_noise)


def create_raw_ica(
    n_channels=20,
    sfreq=250,
    ch_type="mag",
    n_components=20,
    filter_range=(1, 100),
    ica_method="infomax",
    ntime=None,
):
    """Create a Raw instance and ICA instance for testing."""
    n_times = sfreq * 60 if ntime is None else ntime
    rng = np.random.default_rng()
    data = rng.standard_normal((n_channels, n_times))
    ch_names = [f"MEG {i+1}" for i in range(n_channels)]

    # Create valid channel loc for feature extraction
    channel_locs = rng.standard_normal((n_channels, 3))
    channel_locs[:, 0] += 0.1
    channel_locs[:, 1] += 0.1
    channel_locs[:, 2] += 0.1

    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_type)
    for i, loc in enumerate(channel_locs):
        info["chs"][i]["loc"][:3] = loc

    raw = RawArray(data, info)
    raw.filter(*filter_range)

    # fastica can not converge with the current data
    # so we use infomax in computation
    # but set ica_method after fitting for testing
    ica = ICA(n_components=n_components, method="infomax")
    ica.fit(raw)
    if ica_method != "infomax":
        ica.method = ica_method

    return raw, ica


@pytest.fixture
def raw_ica_valid():
    """Raw instance with valid parameters."""
    raw, ica = create_raw_ica()
    return raw, ica


def test_get_megnet_features(raw_ica_valid):
    """Test whether the function returns the correct features."""
    time_series, topomaps = get_megnet_features(*raw_ica_valid)
    n_components = raw_ica_valid[1].n_components
    n_times = raw_ica_valid[0].times.shape[0]

    assert time_series.shape == (n_components, n_times)
    assert topomaps.shape == (n_components, 120, 120, 3)


@pytest.fixture
def raw_ica_invalid_channel():
    """Raw instance with invalid channel type."""
    raw, ica = create_raw_ica(ch_type="eeg")
    return raw, ica


@pytest.fixture
def raw_ica_invalid_sfreq():
    """Raw instance with invalid sampling frequency."""
    raw, ica = create_raw_ica(sfreq=600)
    return raw, ica


@pytest.fixture
def raw_ica_invalid_time():
    """Raw instance with invalid time points."""
    raw, ica = create_raw_ica(ntime=2500)
    return raw, ica


@pytest.fixture
def raw_ica_invalid_filter():
    """Raw instance with invalid filter range."""
    raw, ica = create_raw_ica(filter_range=(0.1, 100))
    return raw, ica


@pytest.fixture
def raw_ica_invalid_ncomp():
    """Raw instance with invalid number of ICA components."""
    raw, ica = create_raw_ica(n_components=10)
    return raw, ica


@pytest.fixture
def raw_ica_invalid_method():
    """Raw instance with invalid ICA method."""
    raw, ica = create_raw_ica(ica_method="fastica")
    return raw, ica


def test_get_megnet_features_invalid(
    raw_ica_invalid_channel,
    raw_ica_invalid_time,
    raw_ica_invalid_sfreq,
    raw_ica_invalid_filter,
    raw_ica_invalid_ncomp,
    raw_ica_invalid_method,
):
    """Test whether the function raises the correct exceptions."""
    test_cases = [
        (raw_ica_invalid_channel, RuntimeError, "Could not find MEG channels"),
        (
            raw_ica_invalid_time,
            RuntimeError,
            "The provided raw instance has 2500 points.",
        ),
        (
            raw_ica_invalid_sfreq,
            RuntimeWarning,
            "The provided raw instance is not sampled at 250 Hz",
        ),
        (
            raw_ica_invalid_filter,
            RuntimeWarning,
            "The provided raw instance is not filtered between 1 and 100 Hz",
        ),
        (
            raw_ica_invalid_ncomp,
            RuntimeWarning,
            "The provided ICA instance has 10 components",
        ),
        (
            raw_ica_invalid_method,
            RuntimeWarning,
            "The provided ICA instance was fitted with 'fastica'",
        ),
    ]

    for raw_ica_fixture, exc_type, msg in test_cases:
        raw, ica = raw_ica_fixture
        if exc_type is RuntimeError:
            with pytest.raises(exc_type, match=msg):
                get_megnet_features(raw, ica)
        elif exc_type is RuntimeWarning:
            with pytest.warns(exc_type, match=msg):
                get_megnet_features(raw, ica)
