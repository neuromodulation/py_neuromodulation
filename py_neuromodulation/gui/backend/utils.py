import multiprocessing as mp
from ansi_colors import ANSI_COLORS
import logging


def force_terminate_process(
    process: mp.Process, name: str, logger: logging.Logger | None = None
) -> None:
    log = logger.debug if logger else print

    import psutil

    p = psutil.Process(process.pid)
    try:
        log(f"Terminating process {name}")
        for child in p.children(recursive=True):
            log(f"Terminating child process {child.pid}")
            child.terminate()
        p.terminate()
        p.wait(timeout=3)
    except psutil.NoSuchProcess:
        log(f"Process {name} has already exited.")
    except psutil.TimeoutExpired:
        log(f"Forcefully killing {name}...")
        p.kill()


def create_logger(name, color: ANSI_COLORS, level=logging.INFO):
    """Function to set up a logger with color coded output"""
    logger = logging.getLogger(name)
    log_format = (
        f"{color}[%(name)s %(levelname)s (%(asctime)s)]:{ANSI_COLORS.RESET} %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(log_format, "%H:%M:%S"))
    logger.setLevel(level)
    logger.addHandler(stream_handler)

    return logger
