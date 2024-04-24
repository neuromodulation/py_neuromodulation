# logger initialization first to prevent circular import
from .utils import _logging  
logger = _logging.logger

# Bring Stream and DataProcessor classes to top namespace
from .nm_stream_offline import Stream
from .nm_run_analysis import DataProcessor