import json
import os
import sys
from pathlib import Path

import mne
import mne_bids
import numpy as np
import pandas as pd
from scipy import io

_PathLike = str | os.PathLike


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
        return nm_channels
    if nm_channels:
        if not os.path.isfile(nm_channels):
            raise ValueError(
                "PATH_NM_CHANNELS is not a valid file. Got: " f"{nm_channels}"
            )
    return pd.read_csv(nm_channels)


def read_BIDS_data(
    PATH_RUN: _PathLike | mne_bids.BIDSPath,
    BIDS_PATH: _PathLike | None = None,
    datatype: str = "ieeg",
) -> tuple[mne.io.Raw, np.ndarray, int | float, int, list | None, list | None]:
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
    if isinstance(PATH_RUN, mne_bids.BIDSPath):
        bids_path = PATH_RUN
    else:
        entities = mne_bids.get_entities_from_fname(PATH_RUN)
        bids_path = mne_bids.BIDSPath(
            subject=entities["subject"],
            session=entities["session"],
            task=entities["task"],
            run=entities["run"],
            acquisition=entities["acquisition"],
            datatype=datatype,
            root=BIDS_PATH,
        )

    raw_arr = mne_bids.read_raw_bids(bids_path)
    coord_list, coord_names = get_coord_list(raw_arr)
    return (
        raw_arr,
        raw_arr.get_data(),
        raw_arr.info["sfreq"],
        int(raw_arr.info["line_freq"]),
        coord_list,
        coord_names,
    )


def get_coord_list(
    raw: mne.io.BaseRaw,
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
        grid = pd.read_csv(
            os.path.join(
                Path(__file__).parent, "grid_" + grid_str.lower() + ".tsv"
            ),
            sep="\t",
        )
    else:
        grid = pd.read_csv(
            os.path.join(PATH_GRIDS, "grid_" + grid_str.lower() + ".tsv"),
            sep="\t",
        )
    return grid


def get_annotations(
    PATH_ANNOTATIONS: str, PATH_RUN: str, raw_arr: mne.io.RawArray
):

    try:
        annot = mne.read_annotations(
            os.path.join(
                PATH_ANNOTATIONS, os.path.basename(PATH_RUN)[:-5] + ".txt"
            )
        )
        raw_arr.set_annotations(annot)

        # annotations starting with "BAD" are omitted with reject_by_annotations 'omit' param
        annot_data = raw_arr.get_data(reject_by_annotation="omit")
    except FileNotFoundError:
        print("Annotations file could not be found")
        print(
            "expected location: "
            + str(
                os.path.join(
                    PATH_ANNOTATIONS, os.path.basename(PATH_RUN)[:-5] + ".txt"
                )
            )
        )
    return annot, annot_data, raw_arr


def read_plot_modules(
    PATH_PLOT=os.path.join(Path(__file__).absolute().parent, "plots")
):
    """Read required .mat files for plotting

    Parameters
    ----------
    PATH_PLOT : regexp, optional
        path to plotting files, by default
    """

    faces = io.loadmat(os.path.join(PATH_PLOT, "faces.mat"))
    vertices = io.loadmat(os.path.join(PATH_PLOT, "Vertices.mat"))
    grid = io.loadmat(os.path.join(PATH_PLOT, "grid.mat"))["grid"]
    stn_surf = io.loadmat(os.path.join(PATH_PLOT, "STN_surf.mat"))
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


def add_labels(
    features: pd.DataFrame,
    settings: dict,
    nm_channels: pd.DataFrame,
    raw_arr_data: np.ndarray,
    fs: int | float,
) -> pd.DataFrame | None:
    """Given a constructed feature data frame, resample the target labels and add to dataframe

    Parameters
    ----------
    features : pd.DataFrame
        computed feature dataframe
    settings_wrapper : settings.py
        initialized settings used for feature estimation
    raw_arr_data : np.ndarray
        raw data including target

    Returns
    -------
    pd.DataFrame | None
        computed feature dataframe including resampled features
    """
    # resample_label
    ind_label = np.where(nm_channels.target == 1)[0]
    if ind_label.shape[0] == 0:
        print("no target specified")
        return None

    offset_time = settings["segment_length_features_ms"]

    offset_start = np.ceil(offset_time / 1000 * fs).astype(int)
    data = raw_arr_data[ind_label, offset_start:]
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    label_downsampled = data[
        :,
        :: int(np.ceil(fs / settings["sampling_rate_features_hz"])),
    ]

    # and add to df
    if features.shape[0] == label_downsampled.shape[1]:
        for idx, label_ch in enumerate(nm_channels.name[ind_label]):
            features[label_ch] = label_downsampled[idx, :]
    else:
        print("label dimensions don't match, saving downsampled label extra")
    return features


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
    if not os.path.exists(os.path.join(out_path, folder_name)):
        print("Creating output folder: " + str(folder_name))
        os.makedirs(os.path.join(out_path, folder_name))

    dict_sidecar = {"fs": fs, "coords": coords, "line_noise": line_noise}

    save_sidecar(dict_sidecar, out_path, folder_name)
    save_features(df_features, out_path, folder_name)
    save_settings(settings, out_path, folder_name)
    save_nm_channels(nm_channels, out_path, folder_name)


def save_settings(
    settings: dict, path_out: _PathLike, folder_name: str | None = None
) -> None:
    path_out = _pathlike_to_str(path_out)
    if folder_name is not None:
        path_out = os.path.join(
            path_out, folder_name, folder_name + "_SETTINGS.json"
        )

    with open(path_out, "w") as f:
        json.dump(settings, f, indent=4)
    print("settings.json saved to " + path_out)


def save_nm_channels(
    nmchannels: pd.DataFrame,
    path_out: _PathLike,
    folder_name: str | None = None,
) -> None:
    path_out = _pathlike_to_str(path_out)
    if folder_name is not None:
        path_out = os.path.join(
            path_out, folder_name, folder_name + "_nm_channels.csv"
        )
    nmchannels.to_csv(path_out)
    print("nm_channels.csv saved to " + path_out)


def save_features(
    df_features: pd.DataFrame,
    path_out: _PathLike,
    folder_name: str | None = None,
) -> None:
    path_out = _pathlike_to_str(path_out)
    if folder_name is not None:
        path_out = os.path.join(
            path_out, folder_name, folder_name + "_FEATURES.csv"
        )
    df_features.to_csv(path_out)
    print("FEATURES.csv saved to " + str(path_out))


def save_sidecar(
    sidecar: dict, path_out: _PathLike, folder_name: str | None = None
) -> None:
    path_out = _pathlike_to_str(path_out)
    save_general_dict(sidecar, path_out, "_SIDECAR.json", folder_name)


def save_general_dict(
    dict_: dict,
    path_out: _PathLike,
    str_add: str,
    folder_name: str | None = None,
) -> None:
    if folder_name is not None:
        path_out = os.path.join(path_out, folder_name, folder_name + str_add)

    with open(path_out, "w") as f:
        json.dump(
            dict_,
            f,
            default=default_json_convert,
            indent=4,
            separators=(",", ": "),
        )
    print(f"{str_add} saved to " + str(path_out))


def default_json_convert(obj) -> list | int | float:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_numpy().tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError("Not serializable")


def read_sidecar(PATH: str) -> dict:
    with open(PATH + "_SIDECAR.json") as f:
        return json.load(f)


def read_settings(PATH: str) -> dict:
    with open(PATH if ".json" in PATH else PATH + "_SETTINGS.json") as f:
        return json.load(f)


def read_features(PATH: str) -> pd.DataFrame:
    return pd.read_csv(PATH + "_FEATURES.csv", index_col=0)


def read_nm_channels(PATH: str) -> pd.DataFrame:
    return pd.read_csv(PATH + "_nm_channels.csv", index_col=0)


def get_run_list_indir(PATH: str) -> list:
    f_files = []
    for dirpath, _, files in os.walk(PATH):
        for x in files:
            if "FEATURES" in x:
                f_files.append(os.path.basename(dirpath))
    return f_files


def loadmat(filename) -> dict:
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def get_paths_example_data():
    """
    This function should provide RUN_NAME, PATH_RUN, PATH_BIDS, PATH_OUT and datatype for the example
    dataset used in most examples.
    """

    SCRIPT_DIR = os.path.dirname(os.path.abspath(''))
    if os.path.basename(SCRIPT_DIR) == "py_neuromodulation":
        # this check is necessary, so we can also run the script from the root directory
        SCRIPT_DIR = os.path.join(SCRIPT_DIR, "examples")

    sys.path.append(os.path.dirname(SCRIPT_DIR))

    sub = "000"
    ses = "right"
    task = "force"
    run = 3
    datatype = "ieeg"

    # Define run name and access paths in the BIDS format.
    RUN_NAME = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}"

    PATH_RUN = os.path.join(
        (os.path.join(SCRIPT_DIR, "data")),
        f"sub-{sub}",
        f"ses-{ses}",
        datatype,
        RUN_NAME,
    )
    PATH_BIDS = os.path.join(SCRIPT_DIR, "data")

    # Provide a path for the output data.
    PATH_OUT = os.path.join(SCRIPT_DIR, "data", "derivatives")

    return RUN_NAME, PATH_RUN, PATH_BIDS, PATH_OUT, datatype


def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj) -> dict:
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def _pathlike_to_str(path: _PathLike) -> str:
    if isinstance(path, str):
        return path
    return str(path)
