from mne.utils import verbose


@verbose
def label_ica_components(inst, ica, verbose=None):
    """Label ICA components.

    Parameters
    ----------
    inst : Raw | Epochs
        The raw data instance that was used for ICA.
    ica : ICA
        The fitted ICA instance.
    %(verbose)s

    Returns
    -------
    gui : instance of ICAComponentLabeler
        The graphical user interface (GUI) window.
    """
    from qtpy.QtWidgets import QApplication

    from ._label_components import ICAComponentLabeler

    # get application
    app = QApplication.instance()
    if app is None:
        app = QApplication(["ICA Component Labeler"])
    gui = ICAComponentLabeler(inst=inst, ica=ica, verbose=verbose)
    gui.show()
    return gui
