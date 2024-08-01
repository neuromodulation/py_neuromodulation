import multiprocessing as mp
import threading
import os
import signal
import time

import logging

from .app_utils import force_terminate_process, create_logger, ansi_color, ansi_reset

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event
    from .app_backend import PyNMBackend

# Shared memory configuration
ARRAY_SIZE = 1000  # Adjust based on your needs


def run_vite(shutdown_event: "Event", debug: bool = False) -> None:
    """Run Vite in a separate shell"""
    import subprocess

    logger = create_logger(
        "Vite",
        "magenta",
        logging.DEBUG if debug else logging.INFO,
    )

    def output_reader(shutdown_event: "Event", process: subprocess.Popen):
        logger.debug("Initialized output stream")
        color = ansi_color(color="magenta", bright=True, styles=["BOLD"])

        def read_stream(stream, stream_name):
            for line in iter(stream.readline, ""):
                if shutdown_event.is_set():
                    break
                logger.info(f"{color}[{stream_name}]{ansi_reset} {line.strip()}")

        stdout_thread = threading.Thread(
            target=read_stream, args=(process.stdout, "stdout")
        )
        stderr_thread = threading.Thread(
            target=read_stream, args=(process.stderr, "stderr")
        )

        stdout_thread.start()
        stderr_thread.start()

        shutdown_event.wait()

        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)

        logger.debug("Output stream closed")

    # Handle different operating systems
    shutdown_signal = signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT
    subprocess_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0

    process = subprocess.Popen(
        "bun run dev",
        cwd="gui_dev",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess_flags,
        shell=True,
    )

    logging_thread = threading.Thread(
        target=output_reader,
        args=(shutdown_event, process),
    )
    logging_thread.start()

    shutdown_event.wait()  # Wait for shutdown

    logger.debug("Terminating Vite server...")
    process.send_signal(shutdown_signal)

    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        logger.debug("Timeout expired, forcing termination...")
        process.kill()

    logging_thread.join(timeout=3)
    if logging_thread.is_alive():
        logger.debug("Logging thread did not finish in time")

    logger.info("Development server stopped")


def create_backend() -> "PyNMBackend":
    from .app_pynm import PyNMState
    from .app_backend import PyNMBackend

    return PyNMBackend(pynm_state=PyNMState())


def run_backend(
    shutdown_event: "Event",
    debug: bool = False,
    reload: bool = True,
) -> None:
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG

    # Configure logging
    color = ansi_color(color="green", bright=True, styles=["BOLD"])
    log_level = "DEBUG" if debug else "INFO"
    log_config = LOGGING_CONFIG.copy()
    log_config["loggers"]["uvicorn"]["level"] = log_level
    log_config["loggers"]["uvicorn.error"]["level"] = log_level
    log_config["loggers"]["uvicorn.access"]["level"] = log_level
    log_config["formatters"]["default"]["fmt"] = (
        f"{color}[FastAPI %(levelname)s (%(asctime)s)]:{ansi_reset} %(message)s"
    )
    log_config["formatters"]["default"]["datefmt"] = "%H:%M:%S"
    log_config["formatters"]["access"]["fmt"] = (
        f"{color}[FastAPI access (%(asctime)s)]:{ansi_reset} %(message)s"
    )
    log_config["formatters"]["access"]["datefmt"] = "%H:%M:%S"

    # Reload requires passing import string
    app = (
        "py_neuromodulation.gui.backend.app_manager:create_backend"
        if reload
        else create_backend()
    )

    server_config = uvicorn.Config(
        app,
        host="localhost",
        reload=reload,
        factory=True,
        port=50000,
        log_level="debug" if debug else "info",
        log_config=log_config,
    )
    server = uvicorn.Server(server_config)

    server_thread = threading.Thread(target=server.run, name="Server")
    server_thread.start()

    shutdown_event.wait()

    server.should_exit = True

    server_thread.join()


class AppManager:
    LAUNCH_FLAG = "PYNM_RUNNING"

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.shutdown_complete = False

        # Prevent launching multiple instances of the app due to multiprocessing
        # This allows the absence of a main guard in the main script
        self.is_child_process = os.environ.get(self.LAUNCH_FLAG) == "TRUE"
        os.environ[self.LAUNCH_FLAG] = "TRUE"

        # Background tasks
        self.tasks: dict[str, mp.Process] = {}

        # Events for multiprocessing synchronization
        self.ready_event = mp.Event()
        self.restart_event = mp.Event()
        self.shutdown_event = mp.Event()

        # PyNM state
        # TODO: need to find a way to pass the state to the backend
        # self.pynm_state = PyNMState()
        # self.app = PyNMBackend(pynm_state=self.pynm_state)

        # Logging
        self.logger = create_logger(
            "PyNM",
            "yellow",
            logging.DEBUG if self.debug else logging.INFO,
        )

    def _run_app(self) -> None:
        self.logger.info("Starting Vite server...")
        self.tasks["vite"] = mp.Process(
            target=run_vite,
            kwargs={"shutdown_event": self.shutdown_event, "debug": self.debug},
            name="Vite",
        )

        self.logger.info("Starting backend server...")
        self.tasks["backend"] = mp.Process(
            target=run_backend,
            kwargs={
                "debug": self.debug,
                "shutdown_event": self.shutdown_event,
                "reload": True,
            },
            name="Backend",
        )

        for process in self.tasks.values():
            process.start()

    def _terminate_app(self) -> None:
        timeout = 5
        deadline = time.time() + timeout

        self.logger.info("App closed, cleaning up background tasks...")
        self.shutdown_event.set()

        for process_name, process in self.tasks.items():
            remaining_time = max(deadline - time.time(), 0)
            process.join(timeout=remaining_time)

            if process.is_alive():
                self.logger.info(
                    f"{process_name} did not terminate in time. Forcing termination..."
                )
                force_terminate_process(process, process_name, logger=self.logger)

            self.logger.info(f"Process {process.name} terminated.")

        self.shutdown_complete = True
        self.shutdown_event.clear()
        self.logger.info("All background tasks succesfully terminated.")

    def launch(self) -> None:
        if self.is_child_process:
            return

        from .app_window import WebViewWindow

        self._run_app()

        self.logger.info("Starting PyWebView window...")
        # PyWebView window only works from main thread
        window = WebViewWindow(debug=True)
        window.register_event_handler("closed", self._terminate_app)
        window.start()

        while not self.shutdown_complete:
            time.sleep(0.1)

        self.shutdown_complete = False
        self.logger.info("All processes cleaned up. Exiting...")
