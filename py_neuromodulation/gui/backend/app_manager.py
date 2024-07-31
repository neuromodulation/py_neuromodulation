import multiprocessing as mp
import threading
import os
import signal
import time

import logging
from utils import force_terminate_process, create_logger, ANSI_COLORS

from app_backend import PyNMBackend
from app_pynm import PyNMState

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event

DEBUG = True

# Shared memory configuration
ARRAY_SIZE = 1000  # Adjust based on your needs


def run_vite(shutdown_event: "Event") -> None:
    """Run Vite in a separate shell"""
    import subprocess

    global DEBUG
    logger = create_logger(
        "Vite",
        ANSI_COLORS.MAGENTA_BOLD_BRIGHT,
        logging.DEBUG if DEBUG else logging.INFO,
    )

    def output_reader(shutdown_event: "Event", process: subprocess.Popen):
        logger.debug("Initialized output stream")

        def read_stream(stream, stream_name):
            for line in iter(stream.readline, ""):
                if shutdown_event.is_set():
                    break
                logger.info(
                    f"{ANSI_COLORS.MAGENTA_BRIGHT}[{stream_name}]{ANSI_COLORS.RESET} {line.strip()}"
                )

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
    shutdown_signals = {"nt": signal.CTRL_BREAK_EVENT, "posix": signal.SIGINT}
    subprocess_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0

    process = subprocess.Popen(
        "bun run dev",
        cwd="gui_dev",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess_flags,
    )

    logging_thread = threading.Thread(
        target=output_reader,
        args=(shutdown_event, process),
    )
    logging_thread.start()

    shutdown_event.wait()  # Wait for shutdown

    logger.debug("Terminating Vite server...")
    process.send_signal(shutdown_signals[os.name])

    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        logger.debug("Timeout expired, forcing termination...")
        process.kill()

    logging_thread.join(timeout=3)
    if logging_thread.is_alive():
        logger.debug("Logging thread did not finish in time")

    logger.info("Development server stopped")


def create_backend() -> PyNMBackend:
    return PyNMBackend(pynm_state=PyNMState())


def run_backend(
    shutdown_event: "Event",
    debug: bool = False,
    reload: bool = True,
) -> None:
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG
    from ansi_colors import ANSI_COLORS

    # Configure logging
    log_level = "%(levelname)s"
    log_msg = "%(message)s"
    log_time = "%(asctime)s"
    log_info = f"{ANSI_COLORS.GREEN_BOLD_BRIGHT}[FastAPI {log_level} ({log_time})]:{ANSI_COLORS.RESET} {log_msg}"
    log_access = f"{ANSI_COLORS.GREEN_BOLD_BRIGHT}[FastAPI access ({log_time})]:{ANSI_COLORS.RESET} {log_msg}"
    log_config = LOGGING_CONFIG.copy()
    log_config["formatters"]["default"]["fmt"] = log_info
    log_config["formatters"]["default"]["datefmt"] = "%H:%M:%S"
    log_config["formatters"]["access"]["fmt"] = log_access
    log_config["formatters"]["access"]["datefmt"] = "%H:%M:%S"

    # Reload requires passing import string
    app = "app_manager:create_backend" if reload else create_backend()

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
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.shutdown_complete = False

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
            ANSI_COLORS.YELLOW_BOLD_BRIGHT,
            logging.DEBUG if self.debug else logging.INFO,
        )

    def run_app(self) -> None:
        self.logger.info("Starting Vite server...")
        self.tasks["vite"] = mp.Process(
            target=run_vite,
            kwargs={"shutdown_event": self.shutdown_event},
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

    def terminate_app(self) -> None:
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
        self.logger.info("All background tasks succesfully terminated.")
