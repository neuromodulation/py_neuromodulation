# Bring Stream and DataProcessor classes to top namespace
from .nm_stream_offline import Stream
from .nm_run_analysis import DataProcessor

__version__ = "0.1.0.dev1"

from .utils import _logging  # logger initialization
