from .nm_logger import PYNMLogger
from pathlib import PurePath
from importlib.metadata import version

__version__ = version("py_neuromodulation")

# Define constant for py_nm directory
PYNM_DIR = PurePath(__file__).parent

# logger initialization first to prevent circular import
logger = PYNMLogger(__name__)

# Bring Stream and DataProcessor classes to top namespace
from .nm_stream_offline import Stream
from .nm_run_analysis import DataProcessor
