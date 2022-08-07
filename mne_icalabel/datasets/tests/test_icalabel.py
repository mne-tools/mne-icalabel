from mne_icalabel.datasets.icalabel import data_path


def test_dataset():
    """Test that the MNE-ICAlabel testing dataset is available."""
    datapath = data_path()
    assert datapath.is_dir()
    assert (datapath / "iclabel").is_dir()
