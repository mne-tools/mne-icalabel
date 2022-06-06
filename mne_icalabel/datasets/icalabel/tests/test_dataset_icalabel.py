import mne_icalabel


def test_dataset_tapping_group():
    datapath = mne_icalabel.datasets.icalabel.data_path()
    assert datapath.is_dir()
    assert (datapath / "iclabel").is_dir()

    # First pass downloaded, check that second pass of access works
    datapath = mne_icalabel.datasets.icalabel.data_path()
    assert datapath.is_dir()
    assert (datapath / "iclabel").is_dir()
