from .utils import _logging  
from pathlib import PurePath
import importlib.metadata

__version__ = importlib.metadata.version('py_neuromodulation')

# Define constant for py_nm directory
PYNM_DIR = PurePath(__file__).parent

# logger initialization first to prevent circular import
logger = _logging.logger

# Bring Stream and DataProcessor classes to top namespace
from .nm_stream_offline import Stream
from .nm_run_analysis import DataProcessor