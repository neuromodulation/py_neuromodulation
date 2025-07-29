Contribution Guide
==================

Welcome to the contributing guide of py_neuromodulation! We are very happy that you are interested in our project.

In general we recommend placing questions and issues in the `GitHub issue tracker <https://github.com/neuromodulation/py_neuromodulation/issues>`_.

For code formatting we use `ruff <https://docs.astral.sh/ruff/formatter/>`_.

For code development, we recommend using the package manager `uv <https://docs.astral.sh/uv/getting-started/installation/>`_.

To setup the python environment, type

::

    uv python install 3.12
    uv venv


Depending on your operating system, activate the python virtual environment: 

::

    . .venv/bin/activate

And install the pyproject.toml dependencies and sync the environment with

::

    uv sync

To install the documentation dependencies use pip:

::

    uv pip install -e .[docs]


GUI development
---------------

Backend
~~~~~~~


The GUI is setup as a react application with a `FastAPI <https://fastapi.tiangolo.com/>`_ backend with a `uvicorn <https://www.uvicorn.org/>`_ server.
The ``gui/backend/app_manager.py`` is the main entry point for the GUI backend. Here multiple tasks are defined and started using python `multiprocessing`.
In debug mode, `vite <https://vite.dev/>`_ is within ``gui/backend/app_manager.run_vite`` used to serve the frontend. In a subprocess ``bun run dev`` is called.
In production mode ``gui/backend/app_manager.run_backend`` directly serves the frontend from the `gui/frontend` directory, if `debug=False`.
Uvicorn (``gui/backend/app_manager.run_uvicorn``) runs an ASGI server, which deploys the FastAPI backend (``gui/backend/app_backend.PyNMBackend``).
Within the backend, multiple `get` and `post` routes are setup to interact with the frontend.

.. raw:: html

   <br>

A *py_neuromodulation* state object is defined within the ``gui/backend/app_pynm.PyNMState`` class.
The state is initialized at the beginning of each App launch, and the settings and channels of the `nm.stream` are then modified during runtime.
Importantly, a ``stream.backend_interface`` is defined which creates an `asyncio` loop to interact with the `nm.stream` object.
Python `multiprocessing.Queue` objects are used for frontend communication, for starting or stopping the stream, or sending data to the frontend through websockets.
Within the ``gui/backend/app_socket.WebSocketManger``, the computed features and raw values are first serialized through `CBOR <https://cbor2.readthedocs.io/en/latest/usage.html>`_ and then sent to the frontend.

Frontend
~~~~~~~~

The frontend is a `react <https://react.dev/>`_ application, which is stored in the `gui_dev` directory.
There are session, settings and socket `zustand <https://zustand-demo.pmnd.rs/>`_ stores. 
Additionally, different pages and components are defined. `Plotly <https://plotly.com/javascript/>`_ is used for real-time dashboard visualization.

For GUI build, `bun <https://bun.sh/package-manager>`_ can be used to build the package. It is then saved into the `gui/frontend` directory.

Building the GUI in *gui_dev* folder:

::

    bun run build

Additionally, the script ``run_gui.py`` can be modified to be in `dev` mode:

::

    App(dev=True).launch()


.. note:: 
    Antonio Brotons wrote an amazing introduction and motivation behind the GUI development this `GitHub discussion <https://github.com/neuromodulation/py_neuromodulation/issues/351#issuecomment-2199932210>`_.