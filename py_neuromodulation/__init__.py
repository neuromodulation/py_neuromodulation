from .nm_logger import PYNMLogger
logger = PYNMLogger(__name__)

from . import (
    nm_analysis,
    nm_stream_abc,
    nm_cohortwrapper,
    nm_across_patient_decoding,
    nm_stream_offline,
    nm_settings,
    nm_define_nmchannels,
)

from .nm_stream_offline import Stream
from .nm_run_analysis import DataProcessor

__version__ = "0.1.0.dev1"
