py_neuromodulation
==================

Documentation: https://neuromodulation.github.io/py_neuromodulation/

Analyzing neural data can be a troublesome, trial and error prone,
and beginner unfriendly process. *py_neuromodulation* allows using a simple
interface for extraction of established neurophysiological features and includes commonly applied pre -and postprocessing methods.

Only **time series data** with a corresponding **sampling frequency** are required for feature extraction.

The output will be a `pandas.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ including different time-resolved computed features. Internally a **stream** get's initialized,
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


Find here the preprint of **py_neuromodulation** called *"Invasive neurophysiology and whole brain connectomics for neural decoding in patients with brain implants"* [1]_.

The original intention for writing this toolbox was movement decoding from invasive brain signals [2]_.
The application however could be any neural decoding problem.
*py_neuromodulation* offers wrappers around common practice machine learning methods for efficient analysis.

Find the documentation here neuromodulation.github.io/py_neuromodulation/ for example usage and parametrization.

Installation
============

py_neuromodulation requires at least python 3.12. For installation you can use pip:

.. code-block::

    pip install py-neuromodulation

Alternatively you can also clone the pacakge and install it using `uv <https://docs.astral.sh/uv/>`_:

.. code-block::

    uv python install 3.12
    uv venv
    . .venv/bin/activate
    uv sync


Then *py_neuromodulation* can be imported via:

.. code-block::

    import py_neuromodulation as nm

Basic Usage
===========

.. code-block:: python
    
    import py_neuromodulation as nm
    import numpy as np
    
    NUM_CHANNELS = 5
    NUM_DATA = 10000
    sfreq = 1000  # Hz
    sampling_rate_features_hz = 3  # Hz

    data = np.random.random([NUM_CHANNELS, NUM_DATA])

    stream = nm.Stream(sfreq=sfreq, data=data, sampling_rate_features_hz=sampling_rate_features_hz)
    features = stream.run()

Check the `Usage <https://neuromodulation.github.io/py_neuromodulation/usage.html>`_ and `First examples <https://neuromodulation.github.io/py_neuromodulation/auto_examples/index.html>`_ for further introduction.

Contact information
-------------------
For any question or suggestion please find my contact
information at `my GitHub profile <https://github.com/timonmerk>`_.

Journal of Open Source Software publication
-------------------------------------------

reStructuredText:
.. image:: https://joss.theoj.org/papers/10.21105/joss.08258/status.svg
   :target: https://doi.org/10.21105/joss.08258


Contributing guide
------------------
https://neuromodulation.github.io/py_neuromodulation/contributing.html


References
----------

.. [1] Merk, T. et al. *Invasive neurophysiology and whole brain connectomics for neural decoding in patients with brain implants*, `https://doi.org/10.21203/rs.3.rs-3212709/v1` (2023).
.. [2] Merk, T. et al. *Electrocorticography is superior to subthalamic local field potentials for movement decoding in Parkinson’s disease*. Elife 11, e75126, `https://doi.org/10.7554/eLife.75126` (2022).
