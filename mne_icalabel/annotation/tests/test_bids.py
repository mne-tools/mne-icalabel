import os.path as op

import pandas as pd
import pytest
from mne.datasets import testing
from mne.io import read_raw_edf
from mne.preprocessing import ICA
from mne_bids import BIDSPath, write_raw_bids

from mne_icalabel.annotation import mark_component, write_components_tsv

subject_id = "01"
session_id = "01"
run = "01"
acq = "01"
task = "testing"
data_path = op.join(testing.data_path(), "EDF")
raw_fname = op.join(data_path, "test_reduced.edf")


@pytest.fixture(scope="function")
def _tmp_bids_path(tmp_path):
    bids_path = BIDSPath(
        subject=subject_id, session=session_id, run=run, acquisition=acq, task=task, root=tmp_path
    )

    raw = read_raw_edf(raw_fname, verbose=False)
    bids_path = write_raw_bids(raw, bids_path, overwrite=True, verbose=False)
    return bids_path


@pytest.fixture(scope="session")
def _ica():
    raw = read_raw_edf(raw_fname, preload=True)
    raw.filter(l_freq=1, h_freq=100)
    n_components = 5
    ica = ICA(n_components=n_components, method="picard")
    ica.fit(raw)
    return ica


def test_write_channels_tsv(_ica, _tmp_bids_path):
    root = _tmp_bids_path.root
    deriv_root = root / "derivatives" / "ICA"
    deriv_fname = BIDSPath(
        root=deriv_root,
        subject=subject_id,
        session=session_id,
        run=run,
        acquisition=acq,
        task=task,
        suffix="channels",
        extension=".tsv",
    )
    _ica = _ica.copy()
    _ica.labels_["ecg"] = [0]

    write_components_tsv(_ica, deriv_fname)

    assert deriv_fname.fpath.exists()
    expected_json = deriv_fname.copy().update(extension=".json")
    assert expected_json.fpath.exists()

    ch_tsv = pd.read_csv(deriv_fname, sep="\t")
    assert all(status == "good" for status in ch_tsv["status"][1:])
    assert ch_tsv["status"][0] == "bad"
    assert ch_tsv["ic_type"].values[0] == "ecg"


def test_mark_components(_ica, _tmp_bids_path):
    root = _tmp_bids_path.root
    deriv_root = root / "derivatives" / "ICA"
    deriv_fname = BIDSPath(
        root=deriv_root,
        subject=subject_id,
        session=session_id,
        run=run,
        acquisition=acq,
        task=task,
        suffix="channels",
        extension=".tsv",
    )
    write_components_tsv(_ica, deriv_fname)

    # mark components
    with pytest.raises(ValueError, match="not a valid label"):
        mark_component(0, deriv_fname, method="manual", label="heart", author="test")

    mark_component(0, deriv_fname, method="manual", label="heart beat", author="test")
    ch_tsv2 = pd.read_csv(deriv_fname, sep="\t")
    assert ch_tsv2["status"][0] == "bad"
    assert all(status == "good" for status in ch_tsv2["status"][1:])
