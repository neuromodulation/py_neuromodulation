import json
import os
import sys
import numpy as np
import pandas as pd
from bids import BIDSLayout
import mne_bids
import define_M1
import features
import generator
import rereference
import resample
import run_analysis
from settings import test_settings

sys.path.append(
    r'C:\Users\ICN_admin\Documents\py_neuromodulation\examples')


def est_features_run(PATH_RUN, PATH_M1=None) -> None:
    """Start feature estimation by reading settings, creating or reading
    df_M1 file with default rereference function (ECoG CAR; depth LFP bipolar)
    Then save features to csv, settings and df_M1 to settings specified output folder.

    Parameters
    ----------
    PATH_RUN : string
        absolute path to run file
    PATH_M1 : string
        absolute path to df_M1.csv file
    """

    # read settings
    with open('settings.json', encoding='utf-8') as json_file:
        settings = json.load(json_file)

    # test settings
    test_settings(settings, verbose=True)

    # read BIDS data
    entities = mne_bids.get_entities_from_fname(PATH_RUN)
    bids_path = mne_bids.BIDSPath(subject=entities["subject"],
                                  session=entities["session"],
                                  task=entities["task"],
                                  run=entities["run"],
                                  acquisition=entities["acquisition"],
                                  datatype="ieeg", root=settings["BIDS_path"])
    raw_arr = mne_bids.read_raw_bids(bids_path)
    ieeg_raw = raw_arr.get_data()
    fs = int(np.ceil(raw_arr.info["sfreq"]))
    line_noise = int(raw_arr.info["line_freq"])

    # read df_M1 / create M1 if None specified
    df_M1 = pd.read_csv(PATH_M1, sep="\t") \
        if PATH_M1 is not None and os.path.isfile(PATH_M1) \
        else define_M1.set_M1(raw_arr.ch_names, raw_arr.get_channel_types())
    
    ref_here = rereference.RT_rereference(df_M1, split_data=False)

    # optionally reduce timing for faster test completion
    # LIMIT_LOW = 50000
    # LIMIT_HIGH = 65000
    # ieeg_raw = ieeg_raw[:,LIMIT_LOW:LIMIT_HIGH]

    # initialize generator for run function
    gen = generator.ieeg_raw_generator(ieeg_raw, settings, fs)

    # define resampler for faster feature estimation
    resample_ = None
    if settings["methods"]["resample_raw"] is True:
        resample_ = resample.Resample(settings, fs)
        fs_new = settings["resample_raw_settings"]["resample_freq"]
    else:
        fs_new = fs

    ch_names = df_M1['name'].to_numpy()
    feature_idx, = np.where(np.logical_and(np.array((df_M1["used"] == 1)),
                                           np.array((df_M1["target"] == 0))))
    used_chs = ch_names[feature_idx].tolist()

    # initialize feature class from settings
    features_ = features.Features(s=settings, fs=fs_new, line_noise=line_noise,
                                  channels=used_chs)

    # call now run_analysis.py
    df_ = run_analysis.run(gen, features_, settings, ref_here, feature_idx,
                           resample_)

    # resample_label
    ind_label = np.where(df_M1["target"] == 1)[0]
    if ind_label.shape[0] != 0:
        offset_time = max([value[1] for value in settings[
            "bandpass_filter_settings"]["frequency_ranges"].values()])
        offset_start = np.ceil(offset_time/1000 * fs).astype(int)
        dat_ = ieeg_raw[ind_label, offset_start:]
        if dat_.ndim == 1:
            dat_ = np.expand_dims(dat_, axis=0)
        label_downsampled = dat_[:, ::int(np.ceil(fs /
                                 settings["sampling_rate_features"]))]

        # and add to df
        if df_.shape[0] == label_downsampled.shape[1]:
            for idx, label_ch in enumerate(df_M1["name"][ind_label]):
                df_[label_ch] = label_downsampled[idx, :]
        else:
            print("label dimensions don't match, saving downsampled label extra")
    else:
        print("no target specified")

    # create out folder if doesn't exist
    folder_name = os.path.basename(PATH_RUN)[:-5]
    if not os.path.exists(os.path.join(settings["out_path"], folder_name)):
        print("create output folder "+str(folder_name))
        os.makedirs(os.path.join(settings["out_path"], folder_name))

    df_.to_csv(os.path.join(settings["out_path"], folder_name,
                            folder_name+"_FEATURES.csv"))

    # save used settings and coordinates to settings as well
    # this becomes necessary since MNE RawArry Info cannot be saved as json
    settings["sfreq"] = raw_arr.info["sfreq"]
    if raw_arr.get_montage() is not None:
        settings["coord_list"] = np.array(list(dict(raw_arr.get_montage().get_positions()
                                          ["ch_pos"]).values())).tolist()
        settings["coord_names"] = np.array(list(dict(raw_arr.get_montage().get_positions()
                                           ["ch_pos"]).keys())).tolist()

    with open(os.path.join(settings["out_path"], folder_name,
                           folder_name+'_SETTINGS.json'), 'w') as f:
        json.dump(settings, f)

    # save df_M1 as csv
    df_M1.to_csv(os.path.join(settings["out_path"], folder_name,
                              folder_name+"_DF_M1.csv"))
