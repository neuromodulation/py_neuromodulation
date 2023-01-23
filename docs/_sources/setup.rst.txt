py_neuromodulation
==================

Click this button to run the "example_BIDS.ipynb":

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/neuromodulation/py_neuromodulation/HEAD

The py_neuromodulation toolbox allows for real time capable processing of multimodal electrophysiological data. The primary use is movement prediction for `adaptive deep brain stimulation <https://pubmed.ncbi.nlm.nih.gov/30607748/>`_.

Setup
=====

For running this toolbox first create a new virtual conda environment and activate it:

.. code-block::

    conda create -n pynm-test python=3.10
    conda activate pynm-test

Then install the packages listed in the `pyproject.toml`.

.. code-block::

    pip install .[dev]
    pytest -v .


Optionally the ipython kernel can be specified to installed for the pynm-test conda environment:

.. code-block::

    ipython kernel install --user --name=pynm-test

Then py_neuromodulation can be imported via:

.. code-block::

    import py_neuromodulation as py_nm

The main modules include running real time enabled feature preprocessing based on `iEEG BIDS <https://www.nature.com/articles/s41597-019-0105-7>`_ data.

Different features can be enabled/disabled and parametrized in the `https://github.com/neuromodulation/py_neuromodulation/blob/main/pyneuromodulation/nm_settings.json>`_.

The current implementation mainly focuses band power and `sharpwave <https://www.sciencedirect.com/science/article/abs/pii/S1364661316302182>`_ feature estimation.

An example folder with a mock subject and derivate `feature <https://github.com/neuromodulation/py_neuromodulation/tree/main/examples/data>`_ set was estimated.

To run feature estimation given the example BIDS data run in root directory.

.. code-block::

    python examples/example_BIDS.py


This will write a feature_arr.csv and different sidecar files in the 'examples/data/derivatives' folder.

For further documentation view `ParametrizationDefinition <ParametrizationDefinition.html#>`_ for description of necessary parametrization files.
`FeatureEstimationDemo <FeatureEstimationDemo.html#>`_ walks through an example feature estimation and explains sharpwave estimation.
