.. _installation:

Installation
============

py_neurmodulation can be installed by cloning the repo and installing the dependencies in a virtual environment:

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

Demos
-----

This documentation provides some example applications:
* `simulation data demo notebook <first_demo.ipynb>`_
* `ECoG movement decoding notebook using BIDS data <example_BIDS.ipynb>`_
* `grid point across patient decoding notebook <example_gridPointProjection.ipynb>`_
* sharpwave exploration notebook
* RMAP example notebook
