import os
import platform
from pathlib import PurePath
from importlib.metadata import version
from .nm_logger import NMLogger
import matplotlib

matplotlib.use("qtagg")  # Set matplotlib backend to TkAgg (Qt backend crashes)

__version__ = version(__package__)  # get version from pyproject.toml

PYNM_DIR = PurePath(__file__).parent  # Define constant for py_nm directory

logger = NMLogger(__name__)  # logger initialization first to prevent circular import

# Set  environment variable MNE_LSL_LIB (required to import Stream below)
LSL_DICT = {
    "Windows": "lsl.dll",
    "Linux": "liblsl.so.1.16.2",
    "Darwin": "liblsl.1.16.0.dylib",
}
os.environ["MNE_LSL_LIB"] = f"{PYNM_DIR.parent}/liblsl/{LSL_DICT[platform.system()]}"

# Bring Stream and DataProcessor classes to top namespace
from .nm_stream_offline import Stream
from .nm_run_analysis import DataProcessor

