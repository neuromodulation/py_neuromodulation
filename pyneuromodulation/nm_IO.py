import mne_bids
import mne
import numpy as np
import os
import json
import _pickle as cPickle
from scipy import io
import pandas as pd
from pathlib import Path

def read_settings(PATH_SETTINGS: str) -> None:
    with open(PATH_SETTINGS, encoding="utf-8") as json_file:
        return json.load(json_file)

def read_BIDS_data(PATH_RUN, BIDS_PATH):
    """Given a run path and bids data path, read the respective data

    Parameters
    ----------
    PATH_RUN : string
    BIDS_PATH : string

    Returns
    -------
    raw_arr : mne.io.RawArray
    raw_arr_data : np.ndarray
    fs : int
    line_noise : int
    """
    entities = mne_bids.get_entities_from_fname(PATH_RUN)

    bids_path = mne_bids.BIDSPath(
        subject=entities["subject"],
        session=entities["session"],
        task=entities["task"],
        run=entities["run"],
        acquisition=entities["acquisition"],
        datatype="ieeg",
        root=BIDS_PATH,
    )

    raw_arr = mne_bids.read_raw_bids(bids_path)

    return (
        raw_arr,
        raw_arr.get_data(),
        int(np.ceil(raw_arr.info["sfreq"])),
        int(raw_arr.info["line_freq"]),
    )

def read_grid(PATH_GRIDS: str, grid_str: str):
    if not PATH_GRIDS:
        grid = pd.read_csv(os.path.join(Path(__file__).parent, 
                "grid_"+grid_str.name.lower()+".tsv"), sep="\t")
    else:
        grid = pd.read_csv(os.path.join(PATH_GRIDS, "grid_"+grid_str.name.lower()+".tsv"), sep="\t")
    return grid

def get_annotations(PATH_ANNOTATIONS:str, PATH_RUN:str, raw_arr:mne.io.RawArray):

    try:
        annot = mne.read_annotations(os.path.join(PATH_ANNOTATIONS,
                                            os.path.basename(PATH_RUN)[:-5]+".txt"))
        raw_arr.set_annotations(annot)

        # annotations starting with "BAD" are omitted with reject_by_annotations 'omit' param
        annot_data = raw_arr.get_data(reject_by_annotation='omit')
    except FileNotFoundError:
        print("Annotations file could not be found")
        print("expected location: "+str(os.path.join(PATH_ANNOTATIONS,
                                        os.path.basename(PATH_RUN)[:-5]+".txt")))
    return annot, annot_data, raw_arr

def read_plot_modules(PATH_PLOT=os.path.join(
                            Path(__file__).absolute().parent.parent,
                            'plots')):
    """Read required .mat files for plotting

    Parameters
    ----------
    PATH_PLOT : regexp, optional
        path to plotting files, by default
    """

    faces = io.loadmat(os.path.join(PATH_PLOT, 'faces.mat'))
    vertices = io.loadmat(os.path.join(PATH_PLOT, 'Vertices.mat'))
    grid = io.loadmat(os.path.join(PATH_PLOT, 'grid.mat'))['grid']
    stn_surf = io.loadmat(os.path.join(PATH_PLOT, 'STN_surf.mat'))
    x_ver = stn_surf['vertices'][::2, 0]
    y_ver = stn_surf['vertices'][::2, 1]
    x_ecog = vertices['Vertices'][::1, 0]
    y_ecog = vertices['Vertices'][::1, 1]
    z_ecog = vertices['Vertices'][::1, 2]
    x_stn = stn_surf['vertices'][::1, 0]
    y_stn = stn_surf['vertices'][::1, 1]
    z_stn = stn_surf['vertices'][::1, 2]

    return faces, vertices, grid, stn_surf, x_ver, y_ver, \
        x_ecog, y_ecog, z_ecog, x_stn, y_stn, z_stn

def add_labels(df_, settings, nm_channels, raw_arr_data, fs, ):
    """Given a constructed feature data frame, resample the target labels and add to dataframe

    Parameters
    ----------
    df_ : pd.DataFrame
        computed feature dataframe
    settings_wrapper : settings.py
        initialized settings used for feature estimation
    raw_arr_data : np.ndarray
        raw data including target

    Returns
    -------
    df_ : pd.DataFrame
        computed feature dataframe including resampled features
    """
    # resample_label
    ind_label = np.where(nm_channels.target == 1)[0]
    if ind_label.shape[0] != 0:
        offset_time = max([value for value in \
                settings["bandpass_filter_settings"]["segment_lengths"].values()])
        offset_start = np.ceil(offset_time / 1000 * fs).astype(int)
        dat_ = raw_arr_data[ind_label, offset_start:]
        if dat_.ndim == 1:
            dat_ = np.expand_dims(dat_, axis=0)
        label_downsampled = dat_[:,:: int(np.ceil(fs / settings["sampling_rate_features"])),]

        # and add to df
        if df_.shape[0] == label_downsampled.shape[1]:
            for idx, label_ch in enumerate(
                nm_channels.name[ind_label]
            ):
                df_[label_ch] = label_downsampled[idx, :]
        else:
            print("label dimensions don't match, saving downsampled label extra")
    else:
        print("no target specified")

    return df_


def save_features_and_settings(
    df_features, run_analysis, folder_name, out_path, settings, nm_channels, coords, fs, line_noise,
    
):
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
    if not os.path.exists(
        os.path.join(out_path, folder_name)
    ):
        print("Creating output folder: " + str(folder_name))
        os.makedirs(
            os.path.join(out_path, folder_name)
        )

    dict_sidecar = {
        "fs" : fs,
        "coords" : coords,
        "line_noise" : line_noise
    }

    save_sidecar(dict_sidecar, out_path, folder_name)
    save_features(df_features, out_path, folder_name)
    save_settings(settings, out_path, folder_name)
    save_nmchannels(nm_channels, out_path, folder_name)

def save_settings(settings: dict, PATH_OUT: str, folder_name: str = None):
    if folder_name is not None:
        PATH_OUT = os.path.join(PATH_OUT, folder_name, folder_name + "_SETTINGS.json")

    with open(PATH_OUT, "w") as f:
        json.dump(settings, f, indent=4)
    print("settings.json saved to " + str(PATH_OUT))

def save_nmchannels(nmchannels: pd.DataFrame, PATH_OUT: str, folder_name: str = None):
    if folder_name is not None:
        PATH_OUT = os.path.join(PATH_OUT, folder_name, folder_name + "_nm_channels.csv")
    nmchannels.to_csv(PATH_OUT)
    print("nm_channels.csv saved to " + str(PATH_OUT))

def save_features(df_features: pd.DataFrame, PATH_OUT: str, folder_name: str = None):
    if folder_name is not None:
        PATH_OUT = os.path.join(PATH_OUT, folder_name, folder_name + "_FEATURES.csv")
    df_features.to_csv(PATH_OUT)
    print("FEATURES.csv saved to " + str(PATH_OUT))

def default_json_convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')

def save_sidecar(sidecar: dict, PATH_OUT: str, folder_name: str = None):

    if folder_name is not None:
        PATH_OUT = os.path.join(PATH_OUT, folder_name, folder_name + "_SIDECAR.json")

    with open(PATH_OUT,'w') as f:
        json.dump(sidecar, f, default=default_json_convert, indent=4, separators=(',', ': '))
    print("sidecar.json saved to " + str(PATH_OUT))
