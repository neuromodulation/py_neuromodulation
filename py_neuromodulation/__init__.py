import os
import sys
import platform
from pathlib import PurePath
from importlib.metadata import version
from .nm_logger import NMLogger
import matplotlib

if sys.platform.startswith("linux"):
    matplotlib.use("Agg")
else:
    matplotlib.use("qtagg")

__version__ = version(__package__)  # get version from pyproject.toml

PYNM_DIR = PurePath(__file__).parent  # Define constant for py_nm directory

logger = NMLogger(__name__)  # logger initialization first to prevent circular import

# Set  environment variable MNE_LSL_LIB (required to import Stream below)
LSL_DICT = {
    "windows_32bit": "windows/x86/liblsl.1.16.2.dll",
    "windows_64bit": "windows/amd64/liblsl.1.16.2.dll",
    "darwin_i386": "macos/amd64/liblsl.1.16.2.dylib",
    "darwin_arm": "macos/arm64/liblsl.1.16.0.dylib",
    "linux_jammy_32bit": "linux/jammy_x86/liblsl.1.16.2.so",
    "linux_jammy_64bit": "linux/jammy_amd64/liblsl.1.16.2.so",
    "linux_focal_64bit": "linux/focal_amd64/liblsl.1.16.2.so",
    "linux_bionic_64bit": "linux/bionic_amd64/liblsl.1.16.2.so",
    "linux_bookworm_64bit": "linux/bookworm_amd64/liblsl.1.16.2.so",
    "linux_noble_64bit": "linux/noble_amd64/liblsl.1.16.2.so",
}

PLATFORM = platform.system().lower().strip()
match PLATFORM:
    case "windows":
        KEY = PLATFORM + "_" +  platform.architecture()[0]
    case "darwin":
        KEY = PLATFORM + "_" +  platform.processor()
    case "linux":
        KEY = PLATFORM + "_" + platform.freedesktop_os_release()["VERSION_CODENAME"]  + "_" + platform.architecture()[0]
    case _: 
        KEY = ""
        
if KEY in LSL_DICT:
    os.environ["MNE_LSL_LIB"] = str(PYNM_DIR / "liblsl" / LSL_DICT[KEY])

# Bring Stream and DataProcessor classes to top namespace
from .nm_stream_offline import Stream
from .nm_run_analysis import DataProcessor
