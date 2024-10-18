import os.path as op

import numpy as np
import onnxruntime as ort
from mne.io import BaseRaw
from mne.preprocessing import ICA
from numpy.typing import NDArray


def megnet_label_components(
    raw: BaseRaw,
    ica: ICA,
    model_path: str = op.join("assets", "network", "megnet.onnx"),
) -> dict:
    """
    Label the provided ICA components with the MEGnet neural network.

    Parameters
    ----------
    raw : BaseRaw
        The raw MEG data.
    ica : mne.preprocessing.ICA
        The ICA data.
    model_path : str
        Path to the ONNX model file.

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

    assert (
        time_series.shape[0] == topomaps.shape[0]
    ), "The number of time series should match the number of spatial topomaps."
    assert topomaps.shape[1:] == (
        120,
        120,
        3,
    ), "The topomaps should have dimensions [N, 120, 120, 3]."
    assert (
        time_series.shape[1] >= 15000
    ), "The time series must be at least 15000 samples long."

    session = ort.InferenceSession(model_path)
    predictions_vote = _chunk_predicting(session, time_series, topomaps)

    all_labels = ["brain/other", "eye blink", "eye movement", "heart"]
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
    """Predict the labels for each component using
    MEGnet's chunk volte algorithm."""
    predction_vote = []

    for comp_series, comp_map in zip(time_series, spatial_maps):
        time_len = comp_series.shape[0]
        start_times = _get_chunk_start(time_len, chunk_len, overlap_len)

        if start_times[-1] + chunk_len <= time_len:
            start_times.append(time_len - chunk_len)

        chunk_votes = {start: 0 for start in start_times}
        for t in range(time_len):
            in_chunks = (
                [start <= t < start + chunk_len for start in start_times]
            )
            # how many chunks the time point is in
            num_chunks = np.sum(in_chunks)
            for start_time, is_in_chunk in zip(start_times, in_chunks):
                if is_in_chunk:
                    chunk_votes[start_time] += 1.0 / num_chunks

        weighted_predictions = {}
        for start_time in chunk_votes.keys():
            onnx_inputs = {
                session.get_inputs()[0].name:
                    np.expand_dims(comp_map, 0).astype(np.float32),
                session.get_inputs()[1].name:
                    np.expand_dims(
                    np.expand_dims(
                        comp_series[start_time: start_time + chunk_len], 0
                        ), -1).astype(np.float32),
            }
            prediction = session.run(None, onnx_inputs)[0]
            weighted_predictions[start_time] = (
                prediction * chunk_votes[start_time]
            )

        comp_prediction = np.stack(
            list(weighted_predictions.values())
            ).mean(axis=0)
        comp_prediction /= comp_prediction.sum()
        predction_vote.append(comp_prediction)

    return np.stack(predction_vote)


def _get_chunk_start(
    input_len: int, chunk_len: int = 15000, overlap_len: int = 3750
) -> list:
    """
    Calculate start times for time series chunks with overlap.
    """
    start_times = []
    start_time = 0
    while start_time + chunk_len <= input_len:
        start_times.append(start_time)
        start_time += chunk_len - overlap_len
    return start_times


if __name__ == "__main__":
    import mne

    from features import get_megnet_features

    sample_dir = mne.datasets.sample.data_path()
    sample_fname = sample_dir / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(sample_fname).pick_types("mag")
    raw.resample(250)
    raw.filter(1, 100)
    ica = mne.preprocessing.ICA(
        n_components=20, max_iter="auto", method="infomax"
        )
    ica.fit(raw)

    res = megnet_label_components(raw, ica)
