.. _introduction:

Introduction
============

Why py_neuromodulation?
-----------------------

Analzing neural data can be a troublesome, trial and error prone,
and beginner unfriendly process. py_neurmodulation allows using a simple
interface extraction of established features and includes common applied pre -and postprocessing methods.

The approach is quite simple, only two things are required:

1. time series data and
2. a corresponding sampling frequency

The output will be a time resolved dataframe including different feature modalities
that were stream in a simulated real-time format. The following features are included:

* oscillatory: fft, stft or bandpass filtered band power
* temporal waveform shape
* fooof
* mne_connectivity estimates
* Hjorth parameter
* non-linear dynamical estimates
* line length
* and more...

How can those features be used?
-------------------------------

The original intention for writing this toolbox was movement decoding from invasive brain signals[1]_.
The application however could be any neural decoding problem.
py_neuromodulation offers wrappers around common pratice Machine Learning methods to efficiently analyze the estimated features.

References
----------

.. [1] Merk, T. et al. *Electrocorticography is superior to subthalamic local field potentials for movement decoding in Parkinson’s disease*. Elife 11, e75126, `https://doi.org/10.7554/eLife.75126` (2022).

