# %%
import mne
import numpy as np

from mne_icalabel.megnet.label_components import megnet_label_components


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
    print(f"this_atrifact_idx: {this_atrifact_idx}")


# %%
raw_ica = raw_ica()
test_megnet_label_components(raw_ica)
# %%
