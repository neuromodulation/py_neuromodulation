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


# Shared memory configuration
ARRAY_SIZE = 1000  # Adjust based on your needs

SERVER_PORT = 50001
DEV_SERVER_PORT = 54321


def create_backend():
    """Factory function passed to Uvicorn to create the web application instance.

    :return: The web application instance.
    :rtype: PyNMBackend
    """
    from .app_pynm import PyNMState
    from .app_backend import PyNMBackend

    debug = os.environ.get("PYNM_DEBUG", "False").lower() == "true"
    dev = os.environ.get("PYNM_DEV", "True").lower() == "true"
    dev_port = os.environ.get("PYNM_DEV_PORT", str(DEV_SERVER_PORT))

    return PyNMBackend(
        pynm_state=PyNMState(),
        debug=debug,
        dev=dev,
        dev_port=int(dev_port),
    )


def run_vite(
    shutdown_event: "Event",
    debug: bool = False,
    dev_port: int = DEV_SERVER_PORT,
    backend_port: int = SERVER_PORT,
) -> None:
    """Run Vite in a separate shell"""
    import subprocess

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    logger = create_logger(
        "Vite",
        "magenta",
        logging.DEBUG if debug else logging.INFO,
    )

    os.environ["VITE_BACKEND_PORT"] = str(backend_port)

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
        "bun run dev --port " + str(dev_port),
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


def run_uvicorn(
    debug: bool = False, reload=False, server_port: int = SERVER_PORT
) -> None:
    from uvicorn.server import Server
    from uvicorn.config import LOGGING_CONFIG, Config

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

    config = Config(
        app="py_neuromodulation.gui.backend.app_manager:create_backend",
        host="localhost",
        reload=reload,
        factory=True,
        port=server_port,
        log_level="debug" if debug else "info",
        log_config=log_config,
    )

    server = Server(config=config)

    if reload:
        from uvicorn.supervisors import ChangeReload
        from uvicorn._subprocess import get_subprocess

        # Overload the restart method of uvicorn so that is does not kill all of our processes
        # IMPORTANT: This is a hack and prevents shutdown events from triggering when the reloader is used
        class CustomReloader(ChangeReload):
            def restart(self) -> None:
                self.process.terminate()  # Use terminate instead of os.kill
                self.process.join()
                self.process = get_subprocess(
                    config=self.config, target=self.target, sockets=self.sockets
                )
                self.process.start()

        sock = config.bind_socket()
        server = CustomReloader(config, target=server.run, sockets=[sock])

    server.run()


def run_backend(
    shutdown_event: "Event",
    dev: bool = True,
    debug: bool = False,
    reload: bool = True,
    server_port: int = SERVER_PORT,
    dev_port: int = DEV_SERVER_PORT,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Pass create_backend parameters through environment variables
    os.environ["PYNM_DEBUG"] = str(debug)
    os.environ["PYNM_DEV"] = str(dev)
    os.environ["PYNM_DEV_PORT"] = str(dev_port)

    server_process = mp.Process(
        target=run_uvicorn,
        kwargs={"debug": debug, "reload": reload, "server_port": server_port},
        name="Server",
    )
    server_process.start()
    shutdown_event.wait()
    server_process.join()


class AppManager:
    LAUNCH_FLAG = "PYNM_RUNNING"

    def __init__(
        self,
        debug: bool = False,
        dev: bool = True,
        run_in_webview=False,
        server_port=SERVER_PORT,
        dev_port=DEV_SERVER_PORT,
    ) -> None:
        """_summary_

        Args:
            debug (bool, optional): If True, run the app in debug mode, which sets logging level to debug,
                and starts uvicorn, FastAPI, and Vite in debug mode. Defaults to False.
            dev (bool, optional): If True, run the app in development mode, which enables hot
                reloading and runs the frontend in Vite server. If False, run the app in production mode,
                which runs the frontend from the static files in the `frontend` directory. Defaults to True.
            run_in_webview (bool, optional): If True, open a PyWebView window to display the app. Defaults to False.
        """
        self.debug = debug
        self.dev = dev
        self.run_in_webview = run_in_webview
        self.server_port = server_port
        self.dev_port = dev_port

        self._reset()
        # Prevent launching multiple instances of the app due to multiprocessing
        # This allows the absence of a main guard in the main script
        self.is_child_process = os.environ.get(self.LAUNCH_FLAG) == "TRUE"
        os.environ[self.LAUNCH_FLAG] = "TRUE"

        self.logger = create_logger(
            "PyNM",
            "yellow",
            logging.DEBUG if self.debug else logging.INFO,
        )

    def _reset(self) -> None:
        """Reset the AppManager to its initial state."""
        # Flags to track the state of the application
        self.shutdown_complete = False
        self.shutdown_started = False

        # Store background tasks
        self.tasks: dict[str, mp.Process] = {}

        # Events for multiprocessing synchronization
        self.shutdown_event: Event = mp.Event()

    def _terminate_app(self) -> None:
        if self.shutdown_started:
            self.logger.info("Termination already in progress. Skipping.")
            return

        self.shutdown_started = True

        timeout = 10
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

    def _sigint_handler(self, signum, frame):
        if not self.shutdown_started:
            self.logger.info("Received SIGINT. Initiating graceful shutdown...")
            self._terminate_app()
        else:
            self.logger.info("SIGINT received again. Ignoring...")

    def launch(self) -> None:
        if self.is_child_process:
            return

        # Handle keyboard interrupt signals
        signal.signal(signal.SIGINT, self._sigint_handler)
        # signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Create and start the subprocesses
        if self.dev:
            self.logger.info("Starting Vite server...")
            self.tasks["vite"] = mp.Process(
                target=run_vite,
                kwargs={
                    "shutdown_event": self.shutdown_event,
                    "debug": self.debug,
                    "dev_port": self.dev_port,
                    "backend_port": self.server_port,
                },
                name="Vite",
            )

        self.logger.info("Starting backend server...")
        self.tasks["backend"] = mp.Process(
            target=run_backend,
            kwargs={
                "shutdown_event": self.shutdown_event,
                "debug": self.debug,
                "reload": self.dev,
                "dev": self.dev,
                "server_port": self.server_port,
                "dev_port": self.dev_port,
            },
            name="Backend",
        )

        for process in self.tasks.values():
            process.start()

        if self.run_in_webview:
            from .app_window import WebViewWindow

            self.logger.info("Starting PyWebView window...")
            window = WebViewWindow(debug=self.debug)  # Must be called from main thread
            window.register_event_handler("closed", self._terminate_app)
            window.start()  # Start the window, this will block until the window is closed
        else:
            try:
                while not self.shutdown_complete:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass  # The SIGINT handler will take care of termination

        if not self.shutdown_complete:
            self._terminate_app()

        self.logger.info("All processes cleaned up. Exiting...")
