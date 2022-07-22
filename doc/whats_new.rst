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

Version 0.3 (Unreleased)
------------------------

...

Enhancements
~~~~~~~~~~~~

-

Bug
~~~

- Ignore the ``RuntimeWarning`` issued by the grid inteprolation for ICLabel topographic feature by `Mathieu Scheltienne`_ (:gh:`69`)

API
~~~

- Add topographic feature using MNE with `~mne_icalabel.features.get_topomap_array` and `~mne_icalabel.features.get_topomaps` by `Anand Saini`_ and `Mathieu Scheltienne`_ (:gh:`71`)
- Add psd feature using MNE with `~mne_icalabel.features.get_psds` by `Anand Saini`_ (:gh:`73`)

Authors
~~~~~~~

* `Mathieu Scheltienne`_
* `Anand Saini`_

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.inc
