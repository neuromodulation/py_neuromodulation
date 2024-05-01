import os
import platform
from pathlib import PurePath
from importlib.metadata import version
from .nm_logger import NMLogger
import matplotlib

# Set matplotlib backend to TkAgg because it crashes with the Qt5 install we're using for LSL
matplotlib.use("tkagg")


__version__ = version("py_neuromodulation")

# Define constant for py_nm directory
PYNM_DIR = PurePath(__file__).parent


# logger initialization first to prevent circular import
logger = NMLogger(__name__)

# Set  environment variable MNE_LSL_LIB (required to import Stream below)
LSL_DICT = {
    "Windows": PYNM_DIR.parent / "liblsl" / "lsl.dll",
    "Linux": PYNM_DIR.parent / "liblsl" / "liblsl.so.1.16.2",
    "Darwin": PYNM_DIR.parent / "liblsl" / "liblsl.1.16.0.dylib"
}

os.environ["MNE_LSL_LIB"] = str(LSL_DICT[platform.system()])


# Bring Stream and DataProcessor classes to top namespace
from .nm_stream_offline import Stream
from .nm_run_analysis import DataProcessor

