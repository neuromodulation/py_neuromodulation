from pathlib import Path
from py_neuromodulation.utils.types import _PathLike
import logging

INFOFORMAT = "%(name)s:\t%(message)s"
DEBUGFORMAT = "%(asctime)s:%(levelname)s:%(name)s:%(filename)s:%(funcName)s:%(lineno)d:\t%(message)s"

LOG_LEVELS = {
    "DEBUG": (logging.DEBUG, DEBUGFORMAT),
    "INFO": (logging.INFO, INFOFORMAT),
    "WARNING": (logging.WARN, DEBUGFORMAT),
    "ERROR": (logging.ERROR, DEBUGFORMAT),
}


class NMLogger(logging.Logger):
    """
    Subclass of logging.Logger with some extra functionality
    """

    def __init__(self, name: str, level: str = "INFO") -> None:
        super().__init__(name, LOG_LEVELS[level][0])

        self.setLevel(level)

        self._console_handler = logging.StreamHandler()
        self._console_handler.setLevel(level)
        self._console_handler.setFormatter(logging.Formatter(LOG_LEVELS[level][1]))

        self.addHandler(self._console_handler)

    def set_level(self, level: str):
        """
        Set console logging level
        """
        self.setLevel(level)
        self._console_handler.setLevel(level)
        self._console_handler.setFormatter(logging.Formatter(LOG_LEVELS[level][1]))

    def log_to_file(self, path: _PathLike, mode: str = "w"):
        """
        Add file handlers to the logger

        Parameters
        ----------
        path: directory where to save logfiles
        mode : str, ('w', 'a')
            w: overwrite files
            a: append to files
        """

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.debug_file_handler = logging.FileHandler(path / "logfile_pydebug.log")
        self.debug_file_handler.setLevel(logging.DEBUG)
        self.debug_file_handler.setFormatter(logging.Formatter(DEBUGFORMAT))

        self.info_file_handler = logging.FileHandler(
            path / "logfile_pyinfo.log", mode=mode
        )
        self.info_file_handler.setLevel(logging.INFO)
        self.info_file_handler.setFormatter(logging.Formatter(INFOFORMAT))

        self.addHandler(self.info_file_handler)
        self.addHandler(self.debug_file_handler)


######################################
# Logger initialization and settings #
######################################

logger = NMLogger(
    "PyNeuromodulation"
)  # logger initialization first to prevent circular import
