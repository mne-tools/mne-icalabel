import os.path as op

import mne_icalabel


def test_dataset_tapping_group():
    datapath = mne_icalabel.datasets.icalabel.data_path()
    assert op.isdir(datapath)
    assert op.isdir(op.join(datapath, "iclabel"))

    # First pass downloaded, check that second pass of access works
    datapath = mne_icalabel.datasets.icalabel.data_path()
    assert op.isdir(datapath)
    assert op.isdir(op.join(datapath, "iclabel"))
