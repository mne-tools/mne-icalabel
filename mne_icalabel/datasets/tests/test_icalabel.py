from mne_icalabel.datasets.icalabel import data_path


def test_dataset(tmp_path):
    """Test that the MNE-ICAlabel testing dataset is available."""
    # download
    datapath = data_path(path=tmp_path)
    assert datapath.is_dir()
    assert (datapath / "iclabel").is_dir()

     # no download
    datapath = data_path(path=tmp_path)
    assert datapath.is_dir()
    assert (datapath / "iclabel").is_dir()
