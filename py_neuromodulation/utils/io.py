import json
from pathlib import PurePath, Path
from typing import TYPE_CHECKING, cast

import numpy as np

from py_neuromodulation.utils.types import _PathLike
from py_neuromodulation.utils.logging import logger
from py_neuromodulation import PYNM_DIR
from mne.io._read_raw import _get_supported

if TYPE_CHECKING:
    from mne_bids import BIDSPath
    from mne import io as mne_io
    import pandas as pd

MNE_FORMATS = list(_get_supported().keys())


def load_channels(
    channels: "pd.DataFrame | _PathLike",
) -> "pd.DataFrame":
    """Read channels from path or specify via BIDS arguments.
    Necessary parameters are then ch_names (list), ch_types (list), bads (list), used_types (list),
    target_keywords (list) and reference Union[list, str].
    """
    import pandas as pd

    if isinstance(channels, pd.DataFrame):
        return channels

    if not Path(channels).is_file():
        raise ValueError("PATH_CHANNELS is not a valid file. Got: " f"{channels}")

    return pd.read_csv(channels)


def read_BIDS_data(
    PATH_RUN: "_PathLike | BIDSPath",
    line_noise: int = 50,
) -> tuple["mne_io.Raw", np.ndarray, float, int, list | None, list | None]:
    """Given a run path and bids data path, read the respective data

    Parameters
    ----------
    PATH_RUN : path to bids run file
        supported formats: https://bids-specification.readthedocs.io/en/v1.2.1/04-modality-specific-files/04-intracranial-electroencephalography.html#ieeg-recording-data
    line_noise: int, optional
        by default 50

    Returns
    -------
    raw_arr : mne.io.RawArray
    raw_arr_data : np.ndarray
    sfreq : float
    line_noise : int
    coord_list : list | None
    coord_names : list | None
    """

    from mne_bids import read_raw_bids, get_bids_path_from_fname

    bids_path = get_bids_path_from_fname(PATH_RUN)

    raw_arr = read_raw_bids(bids_path)
    coord_list, coord_names = get_coord_list(raw_arr)
    if raw_arr.info["line_freq"] is not None:
        line_noise = int(raw_arr.info["line_freq"])
    else:
        logger.info(
            f"Line noise is not available in the data, using value of {line_noise} Hz."
        )
    return (
        raw_arr,
        raw_arr.get_data(),
        raw_arr.info["sfreq"],
        line_noise,
        coord_list,
        coord_names,
    )


def read_mne_data(
    PATH_RUN: "_PathLike | BIDSPath",
    line_noise: int = 50,
) -> tuple[np.ndarray, float, list[str], list[str], list[str]]:
    """Read data in the mne.io.read_raw supported format.

    Parameters
    ----------
    PATH_RUN : _PathLike | BIDSPath
        Path to mne.io.read_raw supported types https://mne.tools/stable/generated/mne.io.read_raw.html
    line_noise : int, optional
        line noise, by default 50

    Returns
    -------
    raw : mne.io.Raw
    sfreq : float
    ch_names : list[str]
    ch_type : list[str]
    bads : list[str]
    """

    from mne import io as mne_io

    raw_arr = mne_io.read_raw(PATH_RUN)
    sfreq = raw_arr.info["sfreq"]
    ch_names = raw_arr.info["ch_names"]
    ch_types = raw_arr.get_channel_types()
    logger.info(
        "Channel data is read using mne.io.read_raw function. Channel types might not be correct"
        " and set to 'eeg' by default"
    )
    bads = raw_arr.info["bads"]

    if raw_arr.info["line_freq"] is not None:
        line_noise = int(raw_arr.info["line_freq"])
    else:
        logger.info(
            f"Line noise is not available in the data, using value of {line_noise} Hz."
        )

    data = cast(np.ndarray, raw_arr.get_data())
    return data, sfreq, ch_names, ch_types, bads


def get_coord_list(
    raw: "mne_io.BaseRaw",
) -> tuple[list, list] | tuple[None, None]:
    """Return the coordinate list and names from mne RawArray

    Parameters
    ----------
    raw : mne_io.BaseRaw

    Returns
    -------
    coord_list[list, list] | coord_names[None, None]
    """
    montage = raw.get_montage()
    if montage is not None:
        coord_list = np.array(
            list(dict(montage.get_positions()["ch_pos"]).values())
        ).tolist()
        coord_names = np.array(
            list(dict(montage.get_positions()["ch_pos"]).keys())
        ).tolist()
    else:
        coord_list = None
        coord_names = None

    return coord_list, coord_names


def read_grid(PATH_GRIDS: _PathLike | None, grid_str: str) -> "pd.DataFrame":
    """Read grid file from path or PYNM_DIR

    Parameters
    ----------
    PATH_GRIDS : _PathLike | None
        path to grid file, by default None
    grid_str : str
        grid name

    Returns
    -------
    pd.DataFrame
        pd.DataFrame including mni x,y,z coordinates for each grid point
    """
    import pandas as pd

    if PATH_GRIDS is None:
        grid = pd.read_csv(PYNM_DIR / ("grid_" + grid_str.lower() + ".tsv"), sep="\t")
    else:
        grid = pd.read_csv(
            PurePath(PATH_GRIDS, "grid_" + grid_str.lower() + ".tsv"), sep="\t"
        )
    return grid


def get_annotations(PATH_ANNOTATIONS: str, PATH_RUN: str, raw_arr: "mne_io.RawArray"):
    filepath = PurePath(PATH_ANNOTATIONS, PurePath(PATH_RUN).name[:-5] + ".txt")
    from mne import read_annotations

    try:
        annot = read_annotations(filepath)
        raw_arr.set_annotations(annot)

        # annotations starting with "BAD" are omitted with reject_by_annotations 'omit' param
        annot_data = raw_arr.get_data(reject_by_annotation="omit")
    except FileNotFoundError:
        logger.critical(f"Annotations file could not be found: {filepath}")

    return annot, annot_data, raw_arr


def read_plot_modules(
    PATH_PLOT: _PathLike = PYNM_DIR / "plots",
):
    """Read required .mat files for plotting

    Parameters
    ----------
    PATH_PLOT : regexp, optional
        path to plotting files, by default
    """

    faces = loadmat(PurePath(PATH_PLOT, "faces.mat"))
    vertices = loadmat(PurePath(PATH_PLOT, "Vertices.mat"))
    grid = loadmat(PurePath(PATH_PLOT, "grid.mat"))["grid"]
    stn_surf = loadmat(PurePath(PATH_PLOT, "STN_surf.mat"))
    x_ver = stn_surf["vertices"][::2, 0]
    y_ver = stn_surf["vertices"][::2, 1]
    x_ecog = vertices["Vertices"][::1, 0]
    y_ecog = vertices["Vertices"][::1, 1]
    z_ecog = vertices["Vertices"][::1, 2]
    x_stn = stn_surf["vertices"][::1, 0]
    y_stn = stn_surf["vertices"][::1, 1]
    z_stn = stn_surf["vertices"][::1, 2]

    return (
        faces,
        vertices,
        grid,
        stn_surf,
        x_ver,
        y_ver,
        x_ecog,
        y_ecog,
        z_ecog,
        x_stn,
        y_stn,
        z_stn,
    )


def write_csv(df, path_out):
    """
    Function to save Pandas dataframes to disk as CSV using
    PyArrow (almost 10x faster than Pandas)
    Difference with pandas.df.to_csv() is that it does not
    write an index column by default
    """
    from pyarrow import csv, Table

    csv.write_csv(Table.from_pandas(df), path_out)


def save_channels(
    nmchannels: "pd.DataFrame",
    out_dir: _PathLike = "",
    prefix: str = "",
) -> None:
    out_dir = Path.cwd() if not out_dir else Path(out_dir)
    filename = "channels.csv" if not prefix else prefix + "_channels.csv"
    write_csv(nmchannels, out_dir / filename)
    logger.info(f"{filename} saved to {out_dir}")


def save_features(
    df_features: "pd.DataFrame",
    out_dir: _PathLike = "",
    prefix: str = "",
) -> None:
    out_dir = Path.cwd() if not out_dir else Path(out_dir)
    filename = f"{prefix}_FEATURES.csv" if prefix else "_FEATURES.csv"
    write_csv(df_features, out_dir / filename)
    logger.info(f"{filename} saved to {str(out_dir)}")


def save_sidecar(
    sidecar: dict,
    out_dir: _PathLike = "",
    prefix: str = "",
) -> None:
    save_general_dict(sidecar, out_dir, prefix, "_SIDECAR.json")


def save_general_dict(
    dict_: dict,
    out_dir: _PathLike = "",
    prefix: str = "",
    str_add: str = "",
) -> None:
    out_dir = Path.cwd() if not out_dir else Path(out_dir)
    filename = f"{prefix}{str_add}"

    with open(out_dir / filename, "w") as f:
        json.dump(
            dict_,
            f,
            default=default_json_convert,
            indent=4,
            separators=(",", ": "),
        )
    logger.info(f"{filename} saved to {out_dir}")


def default_json_convert(obj) -> list | float:
    import pandas as pd

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_numpy().tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError("Not serializable")


def read_sidecar(PATH: _PathLike) -> dict:
    with open(PurePath(str(PATH) + "_SIDECAR.json")) as f:
        return json.load(f)


def read_features(PATH: _PathLike) -> "pd.DataFrame":
    import pandas as pd

    return pd.read_csv(str(PATH) + "_FEATURES.csv", engine="pyarrow")


def read_channels(PATH: _PathLike) -> "pd.DataFrame":
    import pandas as pd

    return pd.read_csv(str(PATH) + "_channels.csv")


def get_run_list_indir(PATH: _PathLike) -> list:
    from os import walk

    f_files = []
    # for dirpath, _, files in Path(PATH).walk(): # Only works in python >=3.12
    for dirpath, _, files in walk(PATH):
        for x in files:
            if "FEATURES" in x:
                f_files.append(PurePath(dirpath).name)
    return f_files


def loadmat(filename) -> dict:
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    from scipy.io import loadmat as sio_loadmat

    data = sio_loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def get_paths_example_data():
    """
    This function should provide RUN_NAME, PATH_RUN, PATH_BIDS, PATH_OUT and datatype for the example
    dataset used in most examples.
    """

    sub = "testsub"
    ses = "EphysMedOff"
    task = "gripforce"
    run = 0
    datatype = "ieeg"

    # Define run name and access paths in the BIDS format.
    RUN_NAME = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}"

    PATH_BIDS = PYNM_DIR / "data"

    PATH_RUN = PYNM_DIR / "data" / f"sub-{sub}" / f"ses-{ses}" / datatype / RUN_NAME

    # Provide a path for the output data.
    PATH_OUT = PATH_BIDS / "derivatives"

    return RUN_NAME, PATH_RUN, PATH_BIDS, PATH_OUT, datatype


def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    from scipy.io.matlab import mat_struct

    for key in dict:
        if isinstance(dict[key], mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj) -> dict:
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    from scipy.io.matlab import mat_struct

    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def generate_unique_filename(path: _PathLike):
    path = Path(path)

    dir = path.parent
    filename = path.stem
    extension = path.suffix

    counter = 1
    while True:
        new_filename = f"{filename}_{counter}{extension}"
        new_file_path = dir / new_filename
        if not new_file_path.exists():
            return Path(new_file_path)
        counter += 1
