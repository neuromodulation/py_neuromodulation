import mne_bids
import numpy as np
import os
import json
import _pickle as cPickle


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


def add_labels(df_, settings_wrapper, raw_arr_data):
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
    ind_label = np.where(settings_wrapper.nm_channels["target"] == 1)[0]
    if ind_label.shape[0] != 0:
        offset_time = max([value for value in settings_wrapper.settings["bandpass_filter_settings"]["segment_lengths"].values()])
        offset_start = np.ceil(offset_time / 1000 * settings_wrapper.settings["fs"]).astype(int)
        dat_ = raw_arr_data[ind_label, offset_start:]
        if dat_.ndim == 1:
            dat_ = np.expand_dims(dat_, axis=0)
        label_downsampled = dat_[
            :,
            :: int(
                np.ceil(
                    settings_wrapper.settings["fs"]
                    / settings_wrapper.settings["sampling_rate_features"]
                )
            ),
        ]

        # and add to df
        if df_.shape[0] == label_downsampled.shape[1]:
            for idx, label_ch in enumerate(
                settings_wrapper.nm_channels["name"][ind_label]
            ):
                df_[label_ch] = label_downsampled[idx, :]
        else:
            print(
                "label dimensions don't match, saving downsampled label extra"
            )
    else:
        print("no target specified")

    return df_


def save_features_and_settings(
    df_, run_analysis, folder_name, settings_wrapper
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
        os.path.join(settings_wrapper.settings["out_path"], folder_name)
    ):
        print("Creating output folder: " + str(folder_name))
        os.makedirs(
            os.path.join(settings_wrapper.settings["out_path"], folder_name)
        )

    PATH_OUT = os.path.join(
        settings_wrapper.settings["out_path"],
        folder_name,
        folder_name + "_FEATURES.csv",
    )
    df_.to_csv(PATH_OUT)
    print("FEATURES.csv saved to " + str(PATH_OUT))

    # rewrite np arrays to lists for json format
    if "coord" in settings_wrapper.settings:
        settings_wrapper.settings["grid_cortex"] = np.array(
            settings_wrapper.settings["grid_cortex"]
        ).tolist()
        settings_wrapper.settings["grid_subcortex"] = np.array(
            settings_wrapper.settings["grid_subcortex"]
        ).tolist()
        settings_wrapper.settings["coord"]["cortex_right"][
            "positions"
        ] = settings_wrapper.settings["coord"]["cortex_right"][
            "positions"
        ].tolist()
        settings_wrapper.settings["coord"]["cortex_left"][
            "positions"
        ] = settings_wrapper.settings["coord"]["cortex_left"][
            "positions"
        ].tolist()
        settings_wrapper.settings["coord"]["subcortex_right"][
            "positions"
        ] = settings_wrapper.settings["coord"]["subcortex_right"][
            "positions"
        ].tolist()
        settings_wrapper.settings["coord"]["subcortex_left"][
            "positions"
        ] = settings_wrapper.settings["coord"]["subcortex_left"][
            "positions"
        ].tolist()

    PATH_OUT = os.path.join(
        settings_wrapper.settings["out_path"],
        folder_name,
        folder_name + "_SETTINGS.json",
    )
    with open(PATH_OUT, "w") as f:
        json.dump(settings_wrapper.settings, f, indent=4)
    print("settings.json saved to " + str(PATH_OUT))

    PATH_OUT = os.path.join(
        settings_wrapper.settings["out_path"],
        folder_name,
        folder_name + "_nm_channels.csv",
    )
    settings_wrapper.nm_channels.to_csv(PATH_OUT)
    print("nm_channels.csv saved to " + str(PATH_OUT))

    PATH_OUT = os.path.join(
        settings_wrapper.settings["out_path"],
        folder_name,
        folder_name + "_run_analysis.p",
    )
    with open(PATH_OUT, "wb") as output:
        cPickle.dump(run_analysis, output)
    print("run analysis.p saved to " + str(PATH_OUT))
