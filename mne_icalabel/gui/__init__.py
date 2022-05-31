from mne.utils import verbose


@verbose
def label_ica_components(info, trans, aligned_ct, subject=None, subjects_dir=None,
                groups=None, verbose=None):
    """Label ICA components.

    Parameters
    ----------
    %(info_not_none)s
    %(trans_not_none)s
    aligned_ct : path-like | nibabel.spatialimages.SpatialImage
        The CT image that has been aligned to the Freesurfer T1. Path-like
        inputs and nibabel image objects are supported.
    %(subject)s
    %(subjects_dir)s
    groups : dict | None
        A dictionary with channels as keys and their group index as values.
        If None, the groups will be inferred by the channel names. Channel
        names must have a format like ``LAMY 7`` where a string prefix
        like ``LAMY`` precedes a numeric index like ``7``. If the channels
        are formatted improperly, group plotting will work incorrectly.
        Group assignments can be adjusted in the GUI.
    %(verbose)s

    Returns
    -------
    gui : instance of IntracranialElectrodeLocator
        The graphical user interface (GUI) window.
    """
    from ._ieeg_locate_gui import IntracranialElectrodeLocator
    from qtpy.QtWidgets import QApplication
    # get application
    app = QApplication.instance()
    if app is None:
        app = QApplication(["Intracranial Electrode Locator"])
    gui = IntracranialElectrodeLocator(
        info, trans, aligned_ct, subject=subject,
        subjects_dir=subjects_dir, groups=groups, verbose=verbose)
    gui.show()
    return gui
