.. py_neuromodulation documentation master file, created by
   sphinx-quickstart on Sun Apr 18 11:04:51 2021.

Welcome to py_neuromodulation's documentation!
==============================================

The py_neuromodulation toolbox allows for real time capable feature estimation of invasive electrophysiological data.
Have a look at the :ref:`installation`, :ref:`usage` and :ref:`examples`.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   installation
   usage
   examples
   api_documentation

Why py_neuromodulation?
-----------------------

Analyzing neural data can be a troublesome, trial and error prone,
and beginner unfriendly process. *py_neuromodulation* allows using a simple
interface for extraction of established features and includes common applied pre -and postprocessing methods.

Basically only **time series data** with a corresponding **sampling frequency** are required.

The output will be a pandas DataFrame including different time-resolved computed features. Internally a **stream** get's initialized,
which resembles an *online* data-stream that can in theory also be be used with a hardware acquisition system.

The following features are currently included:

* oscillatory: fft, stft or bandpass filtered band power
* `temporal waveform shape <https://www.sciencedirect.com/science/article/pii/S1364661316302182>`_
* `fooof <https://fooof-tools.github.io/fooof/>`_
* `mne_connectivity estimates <https://mne.tools/mne-connectivity/stable/index.html>`_ 
* `Hjorth parameter <https://en.wikipedia.org/wiki/Hjorth_parameters>`_
* `non-linear dynamical estimates <https://nolds.readthedocs.io/en/latest/>`_
* various burst features
* line length 
* and more...

How can those features be used?
-------------------------------

The original intention for writing this toolbox was movement decoding from invasive brain signals [1]_.
The application however could be any neural decoding problem.
*py_neuromodulation* offers wrappers around common practice machine learning methods for efficient analysis.

References
----------

.. [1] Merk, T. et al. *Electrocorticography is superior to subthalamic local field potentials for movement decoding in Parkinson’s disease*. Elife 11, e75126, `https://doi.org/10.7554/eLife.75126` (2022).



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
