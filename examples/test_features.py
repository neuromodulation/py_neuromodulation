import json
import multiprocessing
import os
import sys
sys.path.append(
    r'C:\Users\ICN_admin\Documents\py_neuromodulation\pyneuromodulation')

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


def est_features_run(PATH_RUN) -> None:
    #PATH_RUN = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Beijing\sub-FOG006\ses-EphysMedOn\ieeg\sub-FOG006_ses-EphysMedOn_task-ButtonPress_acq-StimOff_run-01_ieeg.vhdr"

    #PATH_M1 = r'C:\Users\ICN_admin\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\Datasets\BIDS_Berlin\derivatives\sub-002\ses-20200131\ieeg\sub-002_ses-20200131_task-SelfpacedRotationR_acq-MedOn+StimOff_run-4_channels_M1.tsv'
    PATH_M1 = None 

    # read settings 
    with open('settings.json', encoding='utf-8') as json_file:
        settings = json.load(json_file)

    #PATH_OUT = settings["out_path"]

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
    
    df_M1 = pd.read_csv(PATH_M1, sep="\t") \
        if PATH_M1 is not None and os.path.isfile(PATH_M1) \
        else define_M1.set_M1(raw_arr.ch_names, raw_arr.get_channel_types())

    ch_names = list(df_M1['name'])
    refs = df_M1['rereference']
    #to_ref_idx = np.array(df_M1[(df_M1['target'] == 0) & (df_M1['used'] == 1)
     #                           & (df_M1["rereference"] != "None")].index)

    to_ref_idx = np.array(df_M1[(df_M1['used'] == 1)].index)

    cortex_idx = np.where(df_M1.ECOG == 1)[0]
    subcortex_idx = np.array(df_M1[(df_M1["ECOG"] == 0) & (df_M1['used'] == 1)
                                   & (df_M1['target'] == 0)].index)

    ref_here = rereference.RT_rereference(ch_names, refs, to_ref_idx,
                                          cortex_idx, subcortex_idx,
                                          split_data=False)

    LIMIT_LOW = 50000
    LIMIT_HIGH = 65000
    #ieeg_raw = ieeg_raw[:,LIMIT_LOW:LIMIT_HIGH]

    gen = generator.ieeg_raw_generator(ieeg_raw, settings, fs)

    feature_idx = np.where(np.logical_and(np.array((df_M1["used"] == 1)),
                                          np.array((df_M1["target"] == 0))))[0]

    resample_ = None
    if settings["methods"]["resample_raw"] is True:
        resample_ = resample.Resample(settings, fs)
        fs_new = settings["resample_raw_settings"]["resample_freq"]

    features_ = features.Features(s=settings, fs=fs_new, line_noise=line_noise,
        channels=np.array(ch_names)[feature_idx])

    # call now run_analysis.py
    df_ = run_analysis.run(gen, features_, settings, ref_here, feature_idx,
                           resample_)

    # resample_label
    ind_label = np.where(df_M1["target"] == 1)[0]
    offset_time = max([value[1] for value in settings[
        "bandpass_filter_settings"]["frequency_ranges"].values()])
    offset_start = np.ceil(offset_time/1000 * fs).astype(int)
    dat_ = ieeg_raw[ind_label, offset_start:]
    if dat_.ndim == 1:
        dat_ = np.expand_dims(dat_, axis=0)
    label_downsampled = dat_[:,
                        ::int(np.ceil(fs / settings["sampling_rate_features"]))]

    # and add to df 
    if df_.shape[0] == label_downsampled.shape[1]:
        for idx, label_ch in enumerate(df_M1["name"][ind_label]):
            df_[label_ch] = label_downsampled[idx, :]
    else: 
        print("label dimensions don't match, saving downsampled label extra")

    # create out folder if doesn't exist 
    folder_name = os.path.basename(PATH_RUN)[:-5]
    if not os.path.exists(os.path.join(settings["out_path"], folder_name)):
        os.makedirs(os.path.join(settings["out_path"], folder_name))

    df_.to_pickle(os.path.join(settings["out_path"], folder_name,
                               folder_name+"_FEATURES.p"))
    
    # save used settings and M1 df as well 
    with open(os.path.join(settings["out_path"], folder_name,
                           folder_name+'_SETTINGS.json'), 'w') as f:
        json.dump(settings, f)

    # save used df_M1 file 
    df_M1.to_pickle(os.path.join(settings["out_path"], folder_name,
                                 folder_name+"_DF_M1.p"))


if __name__ == "__main__":

    PATH_BIDS = "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\Data\\Beijing"
    layout = BIDSLayout(PATH_BIDS)
    run_files = layout.get(extension='.vhdr')
    est_features_run(run_files[0])
    #pool = multiprocessing.Pool()
    #pool.map(est_features_run, run_files)
