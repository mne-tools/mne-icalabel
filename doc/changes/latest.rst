.. NOTE: we use cross-references to highlight new functions and classes.
   Please follow the examples below, so the changelog page will have a link to
   the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Enhancements" for new features
   - "Bugs" for bug fixes
   - "API changes" for backward-incompatible changes

.. NOTE: You can use the :pr:`xx` and :issue:`xx` role to x-ref to a GitHub PR
   or issue from this project.

.. include:: ./authors.inc

.. _latest:

Version 0.10
============

- Fix a spurious warning from :func:`~mne_icalabel.iclabel.get_iclabel_features` (and :func:`~mne_icalabel.label_components`) about the data not being referenced to a common average reference (CAR) when the average reference was applied as a projection (``set_eeg_reference("average", projection=True)``) rather than directly (:pr:`316` by `Leonardo Scappatura`_)
