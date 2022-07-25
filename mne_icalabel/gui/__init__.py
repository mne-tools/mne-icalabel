from mne.preprocessing import ICA


def label_ica_components(inst, ica: ICA, show: bool = True, block: bool = False):
    """Launch the IC labelling GUI.

    Parameters
    ----------
    inst : : Raw | Epochs
        `~mne.io.Raw` or `~mne.Epochs` instance used to fit the `~mne.preprocessing.ICA` decomposition.
    ica : ICA
        The ICA object fitted on `inst`.
    show : bool
        Show the GUI if True.
    block : bool
        Whether to halt program execution until the figure is closed.

    Returns
    -------
    gui : instance of ICAComponentLabeler
        The graphical user interface (GUI) window.
    """
    from mne.viz.backends._utils import _init_mne_qtapp, _qt_app_exec
    from qtpy.QtWidgets import QApplication

    from ._label_components import ICAComponentLabeler

    # get application
    app = _init_mne_qtapp()
    if app is None:
        app = QApplication(["ICA Component Annotator"])
    gui = ICAComponentLabeler(inst=inst, ica=ica, show=show)
    if block:
        _qt_app_exec(app)
    return gui
