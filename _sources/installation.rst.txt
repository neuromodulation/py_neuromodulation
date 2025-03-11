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