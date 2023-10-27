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

Version 0.5
===========

- Compatibility with MNE 1.6 (:pr:`136`, :pr:`140` by `Mathieu Scheltienne`_)
- Raise error if EEG channels are missing from the instance provided to ``ICLABEL`` (:pr:`124` by `Mathieu Scheltienne`_ and `Adam Li`_)
- Add ONNX backend support to ``ICLABEL`` (:pr:`129` by `Jacob Feitelberg`_)
