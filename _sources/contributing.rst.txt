Contribution Guide
==================

Welcome to the contributing guide of py_neuromodulation! We are very happy that you are interested in our project.

In general we recommend placing questions and issues in the `GitHub issue tracker <https://github.com/neuromodulation/py_neuromodulation/issues>`_.

For code formatting we use `ruff <https://docs.astral.sh/ruff/formatter/>`_.

For code development, we recommend using the package manager `uv <https://docs.astral.sh/uv/getting-started/installation/>`_.

To setup the python environment, type

::

    uv python install 3.11
    uv venv


Depending on your operating system, activate the rye virtual environment: 

::

    . .venv/bin/activate

And install the pyproject.toml dependencies:

:: 

    then sync the environment with

::

    uv sync

To install the documentation dependencies use pip:

::

    uv pip install -e .[docs]


