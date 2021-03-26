import sys
sys.path.append(r'C:\Users\ICN_admin\Documents\py_neuromodulation\pyneuromodulation')
import features 
import numpy as np 
import json
import mne_bids
import run_analysis
import generator
import pandas as pd
import define_M1
import os

if __name__ == "__main__":

    PATH_RUN = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Beijing\sub-FOG006\ses-EphysMedOn\ieeg\sub-FOG006_ses-EphysMedOn_task-ButtonPress_acq-StimOff_run-01_ieeg.vhdr"
    #PATH_M1 = r'C:\Users\ICN_admin\Charité - Universitätsmedizin Berlin\Interventional Cognitive Neuromodulation - Data\Datasets\BIDS_Berlin\derivatives\sub-002\ses-20200131\ieeg\sub-002_ses-20200131_task-SelfpacedRotationR_acq-MedOn+StimOff_run-4_channels_M1.tsv'
    PATH_M1 = None 

    # read settings 
    with open('settings.json', encoding='utf-8') as json_file:
        settings = json.load(json_file)

    entities = mne_bids.get_entities_from_fname(PATH_RUN)
    bids_path = mne_bids.BIDSPath(subject=entities["subject"], session=entities["session"], task=entities["task"], \
        run=entities["run"], acquisition=entities["acquisition"], datatype="ieeg", root=settings["BIDS_path"])
    raw_arr = mne_bids.read_raw_bids(bids_path)
    ieeg_raw = raw_arr.get_data()
    fs = int(np.ceil(raw_arr.info["sfreq"]))
    line_noise = int(raw_arr.info["line_freq"])
    
    df_M1 = pd.read_csv(PATH_M1, sep="\t") if PATH_M1 is not None and os.path.isfile(PATH_M1) \
        else define_M1.set_M1(raw_arr.ch_names, raw_arr.get_channel_types())

    ch_names = list(df_M1[(df_M1["used"] == 1) & (df_M1["target"] == 0)]["name"])

    refs = df_M1["rereference"]
    rest_idx = np.array(df_M1[(df_M1["target"] == 0) & (df_M1["used"] == 1)].index)
    cortex_idx = np.where(df_M1.ECOG == 1)[0]
    subcortex_idx = np.array(df_M1[(df_M1["ECOG"] == 0) & (df_M1["used"] == 1) & (df_M1["target"] == 0)].index)
    used = np.where(df_M1.used == 1)[0]
    #LIMIT_LOW = 15000
    #LIMIT_HIGH = 25000
    #ieeg_raw_lim = ieeg_raw[:,LIMIT_LOW:LIMIT_HIGH]

    gen = generator.ieeg_raw_generator(ieeg_raw, df_M1, settings, fs) # clip for timing reasons 

    features_ = features.Features(s=settings, fs=fs, line_noise=line_noise, channels=ch_names)

    # call now run_analysis.py 
    df_ = run_analysis.run(gen, features_, settings, ch_names, refs, rest_idx,\
        cortex_idx, subcortex_idx, used)

    #resample_label 
    ind_label = np.where(df_M1["target"] == 1)[0]
    dat_ = ieeg_raw[ind_label, int(fs*settings["bandpass_filter_settings"]["segment_lengths"][0]):]
    label_downsampled = dat_[:, ::int(np.ceil(fs / settings["resampling_rate"]))]

    # and add to df 
    if df_.shape[0] == label_downsampled.shape[1]:
        for idx, label_ch in enumerate(df_M1["name"][ind_label]):
            df_[label_ch] = label_downsampled[idx, :]
    else: 
        print("label dimensions don't match, saving downsampled label extra")

    df_.to_pickle(os.path.join(settings["out_path"], os.path.basename(PATH_RUN)+"_FEATURES.p"))
    # save used settings and M1 df as well 
    with open(os.path.join(settings["out_path"], os.path.basename(PATH_RUN)+'_SETTINGS.json'), 'w') as f:
        json.dump(settings, f)

    df_M1.to_pickle(settings["out_path"]+"df_M1.py")
    
    