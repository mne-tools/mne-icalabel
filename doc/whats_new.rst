:orphan:

.. _whats_new:


What's new?
===========

Here we list a changelog of MNE-ICALabel.

.. contents:: Contents
   :local:
   :depth: 3

.. currentmodule:: mne_icalabel

.. _current:

Version 0.2 (Unreleased)
------------------------

...

Enhancements
~~~~~~~~~~~~

- Add functions for annotating and labeling ICA components in BIDS format :func:`mne_icalabel.annotation.write_components_tsv`, :func:`mne_icalabel.annotation.mark_component` by `Adam Li`_ (:gh:`60`)

Bug
~~~

- Fix shape of ``'y_pred_proba'`` output from :func:`mne_icalabel.label_components` by `Mathieu Scheltienne`_ (:gh:`36`)
- Add a warning if the ICA decomposition provided does not match the expected decomposition by ``ICLabel``  by `Mathieu Scheltienne`_ (:gh:`42`)
- Fix extraction of PSD feature from ``ICLabel`` model on epochs by `Mathieu Scheltienne`_ (:gh:`64`)
- Fix ICLabel topographic features on ICA fitted with a channel selection performed by ``picks`` by `Mathieu Scheltienne`_ (:gh:`xx`)

API
~~~

-

Authors
~~~~~~~

* `Mathieu Scheltienne`_
* `Adam Li`_

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.inc
