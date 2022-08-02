import pytest
from mne.datasets import testing
from mne.io import read_raw_edf
from mne.preprocessing import ICA

from mne_icalabel.gui import label_ica_components
from mne_icalabel.utils._testing import requires_version

raw = read_raw_edf(testing.data_path() / "EDF" / "test_reduced.edf", preload=True)
raw.filter(l_freq=1, h_freq=100)
ica = ICA(n_components=15, random_state=12345)
ica.fit(raw)


@requires_version("mne", "1.1")
def test_label_components_gui_display():
    ica_ = ica.copy()
    gui = label_ica_components(raw, ica_, show=False)
    # test setting the label
    assert gui.inst == raw
    assert gui.ica == ica
    assert gui.n_components_ == ica.n_components_
    # the initial component should be 0
    assert gui.selected_component == 0


@requires_version("mne", "1.1")
def test_invalid_arguments():
    """Test error error raised with invalid arguments."""
    with pytest.raises(TypeError, match="ica must be an instance of ICA"):
        label_ica_components(raw, 101)

    with pytest.raises(TypeError, match="inst must be an instance of raw or epochs"):
        label_ica_components(101, ica)

    with pytest.raises(ValueError, match="ICA instance should be fitted"):
        label_ica_components(raw, ICA(n_components=10, random_state=12345))
