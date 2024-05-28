import json
from pathlib import PurePath, Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from py_neuromodulation.nm_types import _PathLike
from py_neuromodulation import logger, PYNM_DIR

if TYPE_CHECKING:
    from mne_bids import BIDSPath
    from mne import io as mne_io


def load_nm_channels(
    nm_channels: pd.DataFrame | _PathLike,
) -> pd.DataFrame:
    """Read nm_channels from path or specify via BIDS arguments.
    Nexessary parameters are then
    ch_names (list),
    ch_types (list),
    bads (list)
    used_types (list)
    target_keywords (list)
    reference Union[list, str]
    """

    if isinstance(nm_channels, pd.DataFrame):
        nm_ch_return = nm_channels
    elif nm_channels:
        if not Path(nm_channels).is_file():
            raise ValueError(
                "PATH_NM_CHANNELS is not a valid file. Got: " f"{nm_channels}"
            )
        nm_ch_return = pd.read_csv(nm_channels)

    return nm_ch_return


def read_BIDS_data(
    PATH_RUN: "_PathLike | BIDSPath",
    BIDS_PATH: _PathLike | None = None, # TODO never accessed - delete or make this useful? -> Need to change examples, tests
    datatype: str = "ieeg", # TODO never accessed - delete? -> Need to change examples, tests
    line_noise: int = 50,
) -> tuple["mne_io.Raw", np.ndarray, float, int, list | None, list | None]:
    """Given a run path and bids data path, read the respective data

    Parameters
    ----------
    PATH_RUN : string
    BIDS_PATH : string
    datatype : string

    Returns
    -------
    raw_arr : mne.io.RawArray
    raw_arr_data : np.ndarray
    fs : int
    line_noise : int
    """

    from mne_bids import read_raw_bids, get_bids_path_from_fname, BIDSPath

    if not isinstance(PATH_RUN, BIDSPath):
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


def get_coord_list(
    raw: "mne_io.BaseRaw",
) -> tuple[list, list] | tuple[None, None]:
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


def read_grid(PATH_GRIDS: _PathLike | None, grid_str: str) -> pd.DataFrame:
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


def save_features_and_settings(
    df_features,
    run_analysis,
    folder_name,
    out_path,
    settings,
    nm_channels,
    coords,
    fs,
    line_noise,
) -> None:
    """save settings.json, nm_channels.csv and features.csv

    Parameters
    ----------
    df_ : pd.Dataframe
        feature dataframe
    run_analysis_ : run_analysis.py object
        This includes all (optionally projected) run_analysis estimated data
        inluding added the resampled labels in features_arr
    folder_name : string
        output path
    settings_wrapper : settings.py object
    """

    # create out folder if doesn't exist
    if not Path(out_path, folder_name).exists():
        logger.info(f"Creating output folder: {folder_name}")
        Path(out_path, folder_name).mkdir(parents=True)

    dict_sidecar = {"fs": fs, "coords": coords, "line_noise": line_noise}

    save_sidecar(dict_sidecar, out_path, folder_name)
    save_features(df_features, out_path, folder_name)
    save_settings(settings, out_path, folder_name)
    save_nm_channels(nm_channels, out_path, folder_name)


def write_csv(df, path_out):
    """
    Function to save Pandas dataframes to disk as CSV using
    PyArrow (almost 10x faster than Pandas)
    Difference with pandas.df.to_csv() is that it does not
    write an index column by default
    """
    from pyarrow import csv, Table
    csv.write_csv(Table.from_pandas(df), path_out)


def save_settings(settings: dict, path_out: _PathLike, folder_name: str = "") -> None:
    if folder_name:
        path_out = PurePath(path_out, folder_name, folder_name + "_SETTINGS.json")

    with open(path_out, "w") as f:
        json.dump(settings, f, indent=4)
    logger.info(f"settings.json saved to {path_out}")


def save_nm_channels(
    nmchannels: pd.DataFrame,
    path_out: _PathLike,
    folder_name: str = "",
) -> None:
    if folder_name:
        path_out = PurePath(path_out, folder_name, folder_name + "_nm_channels.csv")
    write_csv(nmchannels, path_out)
    logger.info(f"nm_channels.csv saved to {path_out}")


def save_features(
    df_features: pd.DataFrame,
    path_out: _PathLike,
    folder_name: str = "",
) -> None:
    if folder_name:
        path_out = PurePath(path_out, folder_name, folder_name + "_FEATURES.csv")
    write_csv(df_features, path_out)
    logger.info(f"FEATURES.csv saved to {str(path_out)}")


def save_sidecar(sidecar: dict, path_out: _PathLike, folder_name: str = "") -> None:
    save_general_dict(sidecar, path_out, "_SIDECAR.json", folder_name)


def save_general_dict(
    dict_: dict,
    path_out: _PathLike,
    str_add: str = "",
    folder_name: str = "",
) -> None:
    if folder_name:
        path_out = PurePath(path_out, folder_name, folder_name + str_add)

    with open(path_out, "w") as f:
        json.dump(
            dict_,
            f,
            default=default_json_convert,
            indent=4,
            separators=(",", ": "),
        )
    logger.info(f"{str_add} saved to {path_out}")


def default_json_convert(obj) -> list | float:
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


def read_settings(PATH: _PathLike) -> dict:
    with open(PATH if ".json" in str(PATH) else str(PATH) + "_SETTINGS.json") as f:
        return json.load(f)


def read_features(PATH: _PathLike) -> pd.DataFrame:
    return pd.read_csv(str(PATH) + "_FEATURES.csv", engine="pyarrow")


def read_nm_channels(PATH: _PathLike) -> pd.DataFrame:
    return pd.read_csv(str(PATH) + "_nm_channels.csv")


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
