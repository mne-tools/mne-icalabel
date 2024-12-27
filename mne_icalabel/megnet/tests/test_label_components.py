from unittest.mock import MagicMock

import mne
import numpy as np
import onnxruntime as ort
import pytest
from numpy.testing import assert_allclose

from mne_icalabel.megnet.label_components import (
    _chunk_predicting,
    _get_chunk_start,
    megnet_label_components,
)


@pytest.fixture
def raw_ica():
    """Create a Raw instance and ICA instance for testing."""
    sample_dir = mne.datasets.sample.data_path()
    sample_fname = sample_dir / "MEG" / "sample" / "sample_audvis_raw.fif"

    raw = mne.io.read_raw_fif(sample_fname).pick("mag")
    raw.load_data()
    raw.resample(250)
    raw.notch_filter(60)
    raw.filter(1, 100)

    ica = mne.preprocessing.ICA(n_components=20, method="infomax", random_state=88)
    ica.fit(raw)

    return raw, ica


def test_megnet_label_components(raw_ica):
    """Test whether the function returns the correct artifact index."""
    real_atrifact_idx = [0, 3, 5]  # heart beat, eye movement, heart beat
    prob = megnet_label_components(*raw_ica)
    this_atrifact_idx = [int(idx) for idx in np.nonzero(prob.argmax(axis=1))[0]]
    assert set(real_atrifact_idx) == set(this_atrifact_idx)


def test_get_chunk_start():
    """Test whether the function returns the correct start times."""
    input_len = 10000
    chunk_len = 3000
    overlap_len = 750

    start_times = _get_chunk_start(input_len, chunk_len, overlap_len)

    assert len(start_times) == 4
    assert start_times == [0, 2250, 4500, 6750]


def test_chunk_predicting():
    """Test whether MEGnet's chunk volte algorithm returns the correct shape."""
    rng = np.random.default_rng()
    time_series = rng.random((5, 10000))
    spatial_maps = rng.random((5, 120, 120, 3))

    mock_session = MagicMock(spec=ort.InferenceSession)
    mock_session.run.return_value = [rng.random(4)]

    predictions = _chunk_predicting(
        mock_session, time_series, spatial_maps, chunk_len=3000, overlap_len=750
    )

    assert predictions.shape == (5, 4)
    assert isinstance(predictions, np.ndarray)


def test_ica(raw_ica):
    raw1, ica1 = raw_ica
    raw2 = raw1.copy()
    ica = mne.preprocessing.ICA(n_components=20, method="infomax", random_state=88)
    ica2 = ica.fit(raw2)
    assert_allclose(
        raw1.get_data(),
        raw2.get_data(),
        atol=1e-6,
        err_msg="Raw data should be the same!",
    )

    assert_allclose(
        ica1.mixing_matrix_,
        ica2.mixing_matrix_,
        atol=1e-6,
        err_msg="ICA mixing matrices should be the same!",
    )
    assert_allclose(
        ica1.unmixing_matrix_,
        ica2.unmixing_matrix_,
        atol=1e-6,
        err_msg="ICA unmixing matrices should be the same!",
    )

    ica1_data = ica1.get_sources(raw1).get_data()
    ica2_data = ica2.get_sources(raw2).get_data()
    assert_allclose(
        ica1_data,
        ica2_data,
        atol=1e-6,
        err_msg="ICA transformed data should be the same!",
    )


def test_megnet(raw_ica):
    raw, ica = raw_ica
    prob1 = megnet_label_components(raw, ica)
    prob2 = megnet_label_components(raw, ica)
    assert_allclose(
        prob1, prob2, atol=1e-6, err_msg="MEGnet predictions should be the same!"
    )
