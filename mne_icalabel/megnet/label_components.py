from importlib.resources import files

import numpy as np
import onnxruntime as ort
from mne.io import BaseRaw
from mne.preprocessing import ICA
from numpy.typing import NDArray

from .features import get_megnet_features

_MODEL_PATH: str = files("mne_icalabel.megnet") / "assets" / "megnet.onnx"


def megnet_label_components(raw: BaseRaw, ica: ICA) -> dict:
    """Label the provided ICA components with the MEGnet neural network.

    Parameters
    ----------
    raw : Raw
        Raw MEG recording used to fit the ICA decomposition. The raw instance should be
        bandpass filtered between 1 and 100 Hz and notch filtered at 50 or 60 Hz to
        remove line noise, and downsampled to 250 Hz.
    ica : ICA
        ICA decomposition of the provided instance. The ICA decomposition
        should use the infomax method.

    Returns
    -------
    dict
        Dictionary with the following keys:
            - 'y_pred_proba' : list of float
                The predicted probabilities for each component.
            - 'labels' : list of str
                The predicted labels for each component.

    """
    time_series, topomaps = get_megnet_features(raw, ica)

    # sanity-checks
    assert time_series.shape[0] == topomaps.shape[0]  # number of time-series <-> topos
    assert topomaps.shape[1:] == (120, 120, 3)  # topos are images of shape 120x120x3
    assert 15000 <= time_series.shape[1]  # minimum time-series length

    session = ort.InferenceSession(_MODEL_PATH)
    predictions_vote = _chunk_predicting(session, time_series, topomaps)

    all_labels = ["brain/other", "eye movement", "heart", "eye blink"]
    # megnet_labels = ['NA', 'EB', 'SA', 'CA']
    result = predictions_vote[:, 0, :]
    labels = [all_labels[i] for i in result.argmax(axis=1)]
    proba = [result[i, result[i].argmax()] for i in range(result.shape[0])]

    return {"y_pred_proba": proba, "labels": labels}


def _chunk_predicting(
    session: ort.InferenceSession,
    time_series: NDArray,
    spatial_maps: NDArray,
    chunk_len=15000,
    overlap_len=3750,
) -> NDArray:
    """MEGnet's chunk volte algorithm."""
    predction_vote = []

    for comp_series, comp_map in zip(time_series, spatial_maps):
        time_len = comp_series.shape[0]
        start_times = _get_chunk_start(time_len, chunk_len, overlap_len)

        if start_times[-1] + chunk_len <= time_len:
            start_times.append(time_len - chunk_len)

        chunk_votes = {start: 0 for start in start_times}
        for t in range(time_len):
            in_chunks = [start <= t < start + chunk_len for start in start_times]
            # how many chunks the time point is in
            num_chunks = np.sum(in_chunks)
            for start_time, is_in_chunk in zip(start_times, in_chunks):
                if is_in_chunk:
                    chunk_votes[start_time] += 1.0 / num_chunks

        weighted_predictions = {}
        for start_time in chunk_votes.keys():
            onnx_inputs = {
                session.get_inputs()[0].name: np.expand_dims(comp_map, 0).astype(
                    np.float32
                ),
                session.get_inputs()[1].name: np.expand_dims(
                    np.expand_dims(comp_series[start_time : start_time + chunk_len], 0),
                    -1,
                ).astype(np.float32),
            }
            prediction = session.run(None, onnx_inputs)[0]
            weighted_predictions[start_time] = prediction * chunk_votes[start_time]

        comp_prediction = np.stack(list(weighted_predictions.values())).mean(axis=0)
        comp_prediction /= comp_prediction.sum()
        predction_vote.append(comp_prediction)

    return np.stack(predction_vote)


def _get_chunk_start(
    input_len: int, chunk_len: int = 15000, overlap_len: int = 3750
) -> list:
    """Calculate start times for time series chunks with overlap."""
    start_times = []
    start_time = 0
    while start_time + chunk_len <= input_len:
        start_times.append(start_time)
        start_time += chunk_len - overlap_len
    return start_times
