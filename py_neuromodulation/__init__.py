from .nm_logger import NMLogger
from pathlib import PurePath
from importlib.metadata import version
import matplotlib

# Set matplotlib backend to TkAgg because it crashes with the Qt5 install we're using for LSL
matplotlib.use("tkagg")

__version__ = version("py_neuromodulation")

# Define constant for py_nm directory
PYNM_DIR = PurePath(__file__).parent


# logger initialization first to prevent circular import
logger = NMLogger(__name__)

# Bring Stream and DataProcessor classes to top namespace
from .nm_stream_offline import Stream
from .nm_run_analysis import DataProcessor
