"""Timeflux entry point"""

import logging
import os
import sys
from runpy import run_module
from typing import NoReturn, Sequence, Union

from dotenv import load_dotenv
from timeflux import __version__
from timeflux.core.logging import init_listener, terminate_listener
from timeflux.core.manager import Manager

_LOGGER = logging.getLogger(__name__)


def run(config_file: str) -> None:
    """This code is adapted from `timeflux.py` in the timeflux package."""
    sys.path.append(os.getcwd())
    # args = _args()
    # _init_env(args.env_file, args.env)
    # _init_logging(args.debug)

    _init_env(file = "./.env", vars = None)
    _init_logging("--debug")
    _LOGGER.info("Timeflux %s" % __version__)
    _run_hook("pre")
    try:
        Manager(config_file).run()
        # Manager(args.app).run()
    except Exception as error:
        _LOGGER.error(error)
    _terminate()


def _terminate() -> NoReturn:
    _run_hook("post")
    _LOGGER.info("Terminated")
    terminate_listener()
    sys.exit(0)


def _init_env(file, vars: Union[Sequence[str], None] = None) -> None:
    load_dotenv(file)
    if vars is not None:
        for env in vars:
            if "=" in env:
                env = env.split("=", 1)
                os.environ[env[0]] = env[1]


def _init_logging(debug) -> None:
    level_console = (
        "DEBUG" if debug else os.getenv("TIMEFLUX_LOG_LEVEL_CONSOLE", "INFO")
    )
    level_file = os.getenv("TIMEFLUX_LOG_LEVEL_FILE", "DEBUG")
    file = os.getenv("TIMEFLUX_LOG_FILE", None)
    init_listener(level_console, level_file, file)


def _run_hook(name) -> None:
    module = os.getenv("TIMEFLUX_HOOK_" + name.upper())
    if module:
        _LOGGER.info("Running %s hook" % name)
        try:
            run_module(module)
        except ImportError as error:
            _LOGGER.error(error)
        except Exception as error:
            _LOGGER.exception(error)



if __name__ == "__main__":
    config_file = "motor_stopping.yaml"
    run(config_file)
