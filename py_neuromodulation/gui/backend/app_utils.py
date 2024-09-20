import multiprocessing as mp
import logging
from typing import Sequence
import sys
from pathlib import Path
from py_neuromodulation.utils.types import _PathLike
from functools import lru_cache
import platform


def force_terminate_process(
    process: mp.Process, name: str, logger: logging.Logger | None = None
) -> None:
    log = logger.debug if logger else print

    import psutil

    p = psutil.Process(process.pid)
    try:
        log(f"Terminating process {name}")
        for child in p.children(recursive=True):
            log(f"Terminating child process {child.pid}")
            child.terminate()
        p.terminate()
        p.wait(timeout=3)
    except psutil.NoSuchProcess:
        log(f"Process {name} has already exited.")
    except psutil.TimeoutExpired:
        log(f"Forcefully killing {name}...")
        p.kill()


def create_logger(name, color: str, level=logging.INFO):
    """Function to set up a logger with color coded output"""
    color = ansi_color(color=color, bright=True, styles=["BOLD"])
    logger = logging.getLogger(name)
    log_format = f"{color}[%(name)s %(levelname)s (%(asctime)s)]:{ansi_color(styles=['RESET'])} %(message)s"
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(log_format, "%H:%M:%S"))
    stream_handler.setStream(sys.stderr)
    logger.setLevel(level)
    logger.addHandler(stream_handler)

    return logger


def ansi_color(
    color: str = "DEFAULT",
    bright: bool = True,
    styles: Sequence[str] = [],
    bg_color: str = "DEFAULT",
    bg_bright: bool = True,
) -> str:
    """
    Function to generate ANSI color codes for colored text in the terminal.
    See https://en.wikipedia.org/wiki/ANSI_escape_code

    Returns:
        str: ANSI color code
    """
    ANSI_COLORS = {
        # https://en.wikipedia.org/wiki/ANSI_escape_code
        "BLACK": 30,
        "RED": 31,
        "GREEN": 32,
        "YELLOW": 33,
        "BLUE": 34,
        "MAGENTA": 35,
        "CYAN": 36,
        "WHITE": 37,
        "DEFAULT": 39,
    }

    ANSI_STYLES = {
        "RESET": 0,
        "BOLD": 1,
        "FAINT": 2,
        "ITALIC": 3,
        "UNDERLINE": 4,
        "BLINK": 5,
        "NEGATIVE": 7,
        "CROSSED": 9,
    }

    color = color.upper()
    bg_color = bg_color.upper()
    styles = [style.upper() for style in styles]

    if color not in ANSI_COLORS.keys() or bg_color not in ANSI_COLORS.keys():
        raise ValueError(f"Invalid color: {color}")

    for style in styles:
        if style not in ANSI_STYLES.keys():
            raise ValueError(f"Invalid style: {style}")

    color_code = str(ANSI_COLORS[color] + (60 if bright else 0))
    bg_color_code = str(ANSI_COLORS[bg_color] + 10 + (60 if bg_bright else 0))
    style_codes = ";".join((str(ANSI_STYLES[style]) for style in styles))

    return f"\033[{style_codes};{color_code};{bg_color_code}m"


ansi_reset = ansi_color(styles=["RESET"])


def is_hidden(filepath: _PathLike) -> bool:
    """Check if a file or directory is hidden.

    Args:
        filepath (str): Path to the file or directory.

    Returns:
        bool: True if the file or directory is hidden, False otherwise.
    """
    from pathlib import Path

    filepath = Path(filepath)

    if sys.platform.startswith("win"):
        import ctypes

        try:
            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(filepath))
            assert attrs != -1
            result = bool(attrs & 2) or filepath.name.startswith(".")
        except (AttributeError, AssertionError):
            result = filepath.name.startswith(".")
    else:
        result = filepath.name.startswith(".")

    return result


@lru_cache(maxsize=1)
def get_quick_access():
    system = platform.system()
    if system == "Windows":
        return get_windows_quick_access()
    elif system == "Darwin":  # macOS
        return get_macos_quick_access()
    else:  # Linux, Unix, etc.
        return {"items": []}


def get_windows_quick_access():
    quick_access_items = []

    # Add available drives
    available_drives = [
        f"{d}:\\" for d in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if Path(f"{d}:").exists()
    ]
    for drive in available_drives:
        quick_access_items.append(
            {"name": f"Drive ({drive})", "type": "drive", "path": drive}
        )

    # Get user's pinned folders
    pinned_folders = get_pinned_folders_windows()
    for folder in pinned_folders:
        path = Path(folder["Path"])
        if path.exists():
            quick_access_items.append(
                {"name": folder["Name"], "type": "folder", "path": str(path)}
            )

    # Get user's home directory
    home_path = Path.home()

    # Add common folders if they're not already in pinned folders
    common_folders = [
        ("Desktop", "Desktop"),
        ("Documents", "Documents"),
        ("Downloads", "Downloads"),
        ("Pictures", "Pictures"),
        ("Music", "Music"),
        ("Videos", "Videos"),
    ]

    for folder_name, folder_path in common_folders:
        full_path = home_path / folder_path
        if full_path.exists() and str(full_path) not in [
            item["path"] for item in quick_access_items
        ]:
            quick_access_items.append(
                {"name": folder_name, "type": "folder", "path": str(full_path)}
            )

    # Add user's home directory if not already included
    if str(home_path) not in [item["path"] for item in quick_access_items]:
        quick_access_items.append(
            {"name": "Home", "type": "folder", "path": str(home_path)}
        )

    return {"items": quick_access_items}


def get_pinned_folders_windows():
    import subprocess
    import json

    powershell_command = """
    $shell = New-Object -ComObject Shell.Application
    $quickaccess = $shell.Namespace("shell:::{679f85cb-0220-4080-b29b-5540cc05aab6}").Items()
    $pinned = $quickaccess | Where-Object { $_.IsFolder } | ForEach-Object {
        [PSCustomObject]@{
            Name = $_.Name
            Path = $_.Path
        }
    }
    $pinned | ConvertTo-Json
    """

    try:
        print(powershell_command)
        result = subprocess.run(
            ["powershell", "-Command", powershell_command],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running PowerShell command: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []


def get_macos_quick_access():
    quick_access_folders = get_macos_favorites()
    quick_access_items = []

    quick_access_items.append({"name": "Computer", "type": "drive", "path": "/"})

    # Add Volumes for macOS
    volumes_path = Path("/Volumes")
    if volumes_path.exists():
        for volume in volumes_path.iterdir():
            if volume.is_mount():
                quick_access_items.append(
                    {"name": volume.name, "type": "drive", "path": str(volume)}
                )

    # Add quick access folders
    for folder in quick_access_folders:
        path = Path(folder["Path"])
        if path.exists():
            quick_access_items.append(
                {"name": folder["Name"], "type": "folder", "path": str(path)}
            )

    # Add user's home directory if not already included
    home_path = str(Path.home())
    if home_path not in [item["path"] for item in quick_access_items]:
        quick_access_items.append({"name": "Home", "type": "folder", "path": home_path})

    return {"items": quick_access_items}


def get_macos_favorites():
    import subprocess
    import json

    favorites = []

    try:
        # Common locations in macOS
        common_locations = [
            ("Desktop", Path.home() / "Desktop"),
            ("Documents", Path.home() / "Documents"),
            ("Downloads", Path.home() / "Downloads"),
            ("Pictures", Path.home() / "Pictures"),
            ("Music", Path.home() / "Music"),
            ("Movies", Path.home() / "Movies"),
        ]

        for name, path in common_locations:
            if path.exists():
                favorites.append({"Name": name, "Path": str(path)})

        # Get user-defined favorites from sidebar plist
        plist_path = (
            Path.home()
            / "Library/Application Support/com.apple.sharedfilelist/com.apple.LSSharedFileList.FavoriteItems.sfl2"
        )
        if plist_path.exists():
            try:
                result = subprocess.run(
                    ["plutil", "-convert", "json", "-o", "-", str(plist_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                plist_data = json.loads(result.stdout)
                for item in plist_data.get("Bookmark", []):
                    if "Name" in item and "URL" in item:
                        path = item["URL"].replace("file://", "")
                        favorites.append({"Name": item["Name"], "Path": path})
            except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
                print(f"Error processing macOS favorites: {e}")

    except Exception as e:
        print(f"Error getting macOS favorites: {e}")

    return favorites
