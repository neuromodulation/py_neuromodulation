py_neuromodulation
==================

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

Find the documentation here http://py-neuromodulation.readthedocs.io for example usage and parametrization.

Installation
============

py_neuromodulation requires at least python 3.10. For installation you can use pip:

.. code-block::

    pip install py-neuromodulation


We recommend however installing the package using `rye <https://rye-up.com/guide/installation/>`_:

.. code-block::

    git clone https://github.com/neuromodulation/py_neuromodulation.git
    rye pin 3.11
    rye sync

And then activating the virtual environment e.g. in Windows using:

.. code-block::

    .\.venv\Scripts\activate

Alternatively you can also install the package in a conda environment:

.. code-block::

    conda create -n pynm-test python=3.11
    conda activate pynm-test

Then install the packages listed in the `pyproject.toml`:

.. code-block::

    pip install .


Optionally the ipython kernel can be specified for the installed pynm-test conda environment:

.. code-block::

    ipython kernel install --user --name=pynm-test

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

Check the `Usage <https://py-neuromodulation.readthedocs.io/en/latest/usage.html>`_ and `First examples <https://py-neuromodulation.readthedocs.io/en/latest/auto_examples/plot_0_first_demo.html>`_ for further introduction.

Contact information
-------------------
For any question or suggestion please find my contact
information at `my GitHub profile <https://github.com/timonmerk>`_.

References
----------

.. [1] Merk, T. et al. *Invasive neurophysiology and whole brain connectomics for neural decoding in patients with brain implants*, `https://doi.org/10.21203/rs.3.rs-3212709/v1` (2023).
.. [2] Merk, T. et al. *Electrocorticography is superior to subthalamic local field potentials for movement decoding in Parkinson’s disease*. Elife 11, e75126, `https://doi.org/10.7554/eLife.75126` (2022).
