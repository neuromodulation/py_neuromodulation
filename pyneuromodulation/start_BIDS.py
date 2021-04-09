import json
import os
import sys
import numpy as np
import pandas as pd
from bids import BIDSLayout
import define_M1
import features
import generator
import rereference
import resample
import run_analysis
import settings as nm_settings
import IO

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

    # read and test settings first to obtain BIDS path
    settings_wrapper = nm_settings.SettingsWrapper(settings_path='settings.json')

    # read BIDS data
    raw_arr, raw_arr_data, fs, line_noise = IO.read_BIDS_data(PATH_RUN, settings_wrapper.settings["BIDS_path"])

    # read df_M1 / create M1 if None specified
    settings_wrapper.set_M1(m1_path=None, ch_names=raw_arr.ch_names,
                            ch_types=raw_arr.get_channel_types())
    settings_wrapper.set_fs_line_noise(fs, line_noise)

    # optionally reduce timing for faster test completion
    LIMIT_LOW = 50000
    LIMIT_HIGH = 65000
    raw_arr_data = raw_arr_data[:, LIMIT_LOW:LIMIT_HIGH]

    # initialize generator for run function
    gen = generator.ieeg_raw_generator(raw_arr_data, settings_wrapper.settings)

    # define resampler for faster feature estimation
    ref_here = rereference.RT_rereference(settings_wrapper.df_M1, split_data=False)

    if settings_wrapper.settings["methods"]["resample_raw"] is True:
        resample_ = resample.Resample(settings_wrapper.settings)
    else:
        resample_ = None

    # initialize feature class from settings
    features_ = features.Features(settings_wrapper.settings)

    # call now run_analysis.py
    df_ = run_analysis.run(gen, features_, settings_wrapper.settings, ref_here, resample_)

    # add resampled labels to feature dataframe
    df_ = IO.add_labels(df_, settings_wrapper, raw_arr_data)

    # save settings.json, df_M1.tsv and features.csv
    IO.save_features_and_settings(df_=df_, folder_name=os.path.basename(PATH_RUN)[:-5],
                                  settings_wrapper=settings_wrapper)
