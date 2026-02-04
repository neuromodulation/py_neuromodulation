Installation
============

py_neuromodulation requires at least python 3.12. For installation you can use pip:

.. code-block::

    pip install py-neuromodulation

We recommend, however, using the package manager `uv <https://docs.astral.sh/uv/getting-started/installation/>`_, and setting up a clean virtual environment:

.. code-block::

    git clone https://github.com/neuromodulation/py_neuromodulation.git
    uv python install 3.12
    uv venv

And then activating the virtual environment e.g. in Windows using:

.. code-block::

    .\.venv\Scripts\activate

or in unix-based systems:

.. code-block::

    source .venv/bin/activate

Then install the packages listed in the `pyproject.toml`:

.. code-block::

    uv sync

The GUI can then simply be started by running:

.. code-block::

    run_gui

If needed, install the documentation dependencies:

.. code-block::

    uv pip install -e .[docs]


Alternatively, you can also install the package in a conda environment:

    conda create -n pynm-test python=3.12
    conda activate pynm-test

Then install the packages listed in the `pyproject.toml`:

.. code-block::

    pip install .
