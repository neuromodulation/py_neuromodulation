import multiprocessing as mp
import logging
from typing import Sequence
import sys


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


def create_logger(name, color: str, level=logging.INFO):
    """Function to set up a logger with color coded output"""
    color = ansi_color(color=color, bright=True, styles=["BOLD"])
    logger = logging.getLogger(name)
    log_format = f"{color}[%(name)s %(levelname)s (%(asctime)s)]:{ansi_color(styles=['RESET'])} %(message)s"
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(log_format, "%H:%M:%S"))
    stream_handler.setStream(sys.stderr)
    logger.setLevel(level)
    logger.addHandler(stream_handler)

    return logger


def ansi_color(
    color: str = "DEFAULT",
    bright: bool = True,
    styles: Sequence[str] = [],
    bg_color: str = "DEFAULT",
    bg_bright: bool = True,
) -> str:
    """
    Function to generate ANSI color codes for colored text in the terminal.
    See https://en.wikipedia.org/wiki/ANSI_escape_code

    Returns:
        str: ANSI color code
    """
    ANSI_COLORS = {
        # https://en.wikipedia.org/wiki/ANSI_escape_code
        "BLACK": 30,
        "RED": 31,
        "GREEN": 32,
        "YELLOW": 33,
        "BLUE": 34,
        "MAGENTA": 35,
        "CYAN": 36,
        "WHITE": 37,
        "DEFAULT": 39,
    }

    ANSI_STYLES = {
        "RESET": 0,
        "BOLD": 1,
        "FAINT": 2,
        "ITALIC": 3,
        "UNDERLINE": 4,
        "BLINK": 5,
        "NEGATIVE": 7,
        "CROSSED": 9,
    }

    color = color.upper()
    bg_color = bg_color.upper()
    styles = [style.upper() for style in styles]

    if color not in ANSI_COLORS.keys() or bg_color not in ANSI_COLORS.keys():
        raise ValueError(f"Invalid color: {color}")

    for style in styles:
        if style not in ANSI_STYLES.keys():
            raise ValueError(f"Invalid style: {style}")

    color_code = str(ANSI_COLORS[color] + (60 if bright else 0))
    bg_color_code = str(ANSI_COLORS[bg_color] + 10 + (60 if bg_bright else 0))
    style_codes = ";".join((str(ANSI_STYLES[style]) for style in styles))

    return f"\033[{style_codes};{color_code};{bg_color_code}m"


ansi_reset = ansi_color(styles=["RESET"])
