Contribution Guide
==================

Welcome to the contributing guide of py_neuromodulation! We are very happy that you are interested in our project.

In general we recommend placing questions and issues in the `GitHub issue tracker <https://github.com/neuromodulation/py_neuromodulation/issues>`_.

For code formatting we use `ruff <https://docs.astral.sh/ruff/formatter/>`_.

For code development, we recommend using the package manager `rye <https://rye.astral.sh/>`_.

To setup the python environment, type

::

    rye pin 3.12

then sync the environment with

::

    rye sync

Depending on your operating system, activate the rye virtual environment: 

::

    . .venv/bin/activate

To install the documentation dependencies use pip:

::

    python -m pip install .[docs]


