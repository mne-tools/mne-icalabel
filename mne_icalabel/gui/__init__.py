from mne.utils import verbose


@verbose
def label_ica_components(ica, inst=None, verbose=None):
    """Label ICA components.

    Parameters
    ----------
    ica : ICA
        The fitted ICA instance.
    inst : Raw | Epochs
        The raw data instance that was used for ICA.
    %(verbose)s

    Returns
    -------
    gui : instance of ICAComponentLabeler
        The graphical user interface (GUI) window.
    """
    from qtpy.QtWidgets import QApplication

    from mne_icalabel.gui._label_components import ICAComponentLabeler

    # get application
    app = QApplication.instance()
    if app is None:
        app = QApplication(["ICA Component Labeler"])
    gui = ICAComponentLabeler(inst=inst, ica=ica)
    gui.show()
    return gui


if __name__ == "__main__":
    from mne.datasets import sample
    from mne.io import read_raw
    from mne.preprocessing import ICA

    directory = sample.data_path() / "MEG" / "sample"
    raw = read_raw(directory / "sample_audvis_raw.fif", preload=False)
    raw.crop(0, 10).pick_types(eeg=True, exclude="bads")
    raw.load_data()
    # preprocess
    raw.filter(l_freq=1.0, h_freq=100.0)
    raw.set_eeg_reference("average")

    n_components = 15
    ica = ICA(n_components=n_components, method="picard")
    ica.fit(raw)

    # annotate ICA components
    gui = label_ica_components(ica, raw)
