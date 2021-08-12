import os
from pathlib import Path
import mne

import define_M1
import features
import generator
import rereference
import resample
import run_analysis
import settings as nm_settings
import IO
import projection


def est_features_run(
        PATH_RUN, PATH_M1=None, PATH_SETTINGS=None, 
        PATH_ANNOTATIONS=None, verbose=True) -> None:
    """Start feature estimation by reading settings, creating or reading
    df_M1 file with default rereference function (ECoG CAR; depth LFP bipolar)
    Then save features to csv, settings and df_M1 to settings specified output folder.
    Parameters
    ----------
    PATH_RUN : string
        absolute path to run file
    PATH_M1 : string
        absolute path to df_M1.csv file
    PATH_SETTINGS : string
        absolute path to settings.json file
    PATH_ANNOTATIONS : string
        absolute path to folder with mne annotations.txt
    """

    # read and test settings first to obtain BIDS path
    if PATH_SETTINGS is None:
        PATH_PYNEUROMODULATION = Path(__file__).absolute().parent.parent
        settings_path = os.path.join(
            PATH_PYNEUROMODULATION, 'examples', 'settings.json')
        settings_wrapper = nm_settings.SettingsWrapper(
            settings_path=settings_path)
    else:
        settings_wrapper = nm_settings.SettingsWrapper(
            settings_path=PATH_SETTINGS)

    # read BIDS data
    raw_arr, raw_arr_data, fs, line_noise = IO.read_BIDS_data(
        PATH_RUN, settings_wrapper.settings["BIDS_path"])

    # potentially read annotations
    if PATH_ANNOTATIONS is not None:
        try:
            annot = mne.read_annotations(os.path.join(PATH_ANNOTATIONS, os.path.basename(PATH_RUN)[:-5]+".txt"))
            raw_arr.set_annotations(annot)
            # annotations starting with "BAD" are omitted with reject_by_annotations 'omit' param
            raw_arr_data = raw_arr.get_data(reject_by_annotation='omit')
        except FileNotFoundError:
            print("Annotations file could not be found")
            print("expected location: "+str(os.path.join(PATH_ANNOTATIONS, os.path.basename(PATH_RUN)[:-5]+".txt")))
            pass

    # (if available) add coordinates to settings
    if raw_arr.get_montage() is not None:
        settings_wrapper.add_coord(raw_arr.copy())

    if any((settings_wrapper.settings["methods"]["project_cortex"],
            settings_wrapper.settings["methods"]["project_subcortex"])):
        projection_ = projection.Projection(settings_wrapper.settings)
    else:
        projection_ = None

    # read df_M1 / create M1 if None specified
    settings_wrapper.set_M1(m1_path=PATH_M1, ch_names=raw_arr.ch_names,
                            ch_types=raw_arr.get_channel_types())
    settings_wrapper.set_fs_line_noise(fs, line_noise)

    # optionally reduce timing for faster test completion
    #LIMIT_LOW = 0 # start at 100s 
    #LIMIT_HIGH = 120000# add 20 s data analysis
    #raw_arr_data = raw_arr_data[:, LIMIT_LOW:LIMIT_HIGH]

    # select only ECoG
    settings_wrapper.df_M1.loc[(settings_wrapper.df_M1["type"] == "seeg") |
                               (settings_wrapper.df_M1["type"] == "dbs"),
                               "used"] = 0
    # initialize generator for run function
    gen = generator.ieeg_raw_generator(raw_arr_data, settings_wrapper.settings)

    # initialize rereferencing
    if settings_wrapper.settings["methods"]["re_referencing"] is True:
        rereference_ = rereference.RT_rereference(
            settings_wrapper.df_M1, split_data=False)
    else:
        rereference_ = None
        # reset df_M1 from default values
        settings_wrapper.df_M1["rereference"] = None
        settings_wrapper.df_M1["new_name"] = settings_wrapper.df_M1["name"]

    # define resampler for faster feature estimation
    if settings_wrapper.settings["methods"]["raw_resampling"] is True:
        resample_ = resample.Resample(settings_wrapper.settings)
    else:
        resample_ = None

    # initialize feature class from settings
    features_ = features.Features(settings_wrapper.settings, verbose=verbose)

    # initialize run object
    run_analysis_ = run_analysis.Run(
        features_, settings_wrapper.settings, rereference_, projection_,
        resample_, verbose=verbose)

    while True:
        ieeg_batch = next(gen, None)
        if ieeg_batch is not None:
            run_analysis_.run(ieeg_batch)
        else:
            break

    # add resampled labels to feature dataframe
    df_ = IO.add_labels(
        run_analysis_.feature_arr, settings_wrapper, raw_arr_data)

    # save settings.json, df_M1.tsv and features.csv
    # plus pickled run_analysis including projections
    run_analysis_.feature_arr = df_  # here the potential label stream is added
    IO.save_features_and_settings(df_=df_, run_analysis_=run_analysis_,
                                  folder_name=os.path.basename(PATH_RUN)[:-5],
                                  settings_wrapper=settings_wrapper)