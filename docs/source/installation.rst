.. _installation:

Installation
============

py_neurmodulation can be installed by cloning the repo and installing the dependencies in a virtual environment:

.. code-block::

    git clone https://github.com/neuromodulation/py_neuromodulation.git
    conda create -n pynm-test python=3.11
    conda activate pynm-test

Then install the packages listed in the `pyproject.toml`.

.. code-block::

    pip install .[dev]