.. _introduction:

Introduction
============

Why py_neuromodulation?
-----------------------

Analzing neural data can be a troublesome, trial and error prone,
and beginner unfriendly process. *py_neurmodulation* allows using a simple
interface extraction of established features and includes common applied pre -and postprocessing methods.

Basically only **time series data** with a corresponding **sampling frequency** are required.

The output will be a dataframe including different time-resolved computed features. Internally a **stream** get's initialized,
which resembles an *online* data-stream can be be used with a hardware acquisition system. 

The following features are included:

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
py_neuromodulation offers wrappers around common practise Machine Learning methods for efficient analysis.

References
----------

.. [1] Merk, T. et al. *Electrocorticography is superior to subthalamic local field potentials for movement decoding in Parkinson’s disease*. Elife 11, e75126, `https://doi.org/10.7554/eLife.75126` (2022).

