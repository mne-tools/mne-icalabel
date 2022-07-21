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
~~~~~~~~~~~~`

-

Bug
~~~

- Fix shape of ``'y_pred_proba'`` output from `mne_icalabel.label_components` by `Mathieu Scheltienne`_ (:gh:`36`)
- Add a warning if the ICA decomposition provided does not match the expected decomposition by ``ICLabel``  by `Mathieu Scheltienne`_ (:gh:`42`)

API
~~~

- Add topographic feature using MNE with `~mne_icalabel.features.get_topomap` and `~mne_icalabel.features.get_topomaps` by `Anand Saini`_ (:gh:`71)

Authors
~~~~~~~

* `Mathieu Scheltienne`_
* `Anand Saini`_

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.inc
