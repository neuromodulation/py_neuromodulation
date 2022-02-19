import sys 
import os
from bids import BIDSLayout
from itertools import product
from mne_realtime import LSLClient
import json
import numpy as np
from pathlib import Path

# first parent to get example folder, second py_neuromodulation folder
PATH_PYNEUROMODULATION = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation'))

from settings import test_settings
import define_M1
import rereference
import features
import run_analysis
import generator_LSL
import settings as nm_settings
import projection
import IO
import resample

if __name__ == "__main__":

    # M1 needs to be defined 

    HOST_NAME = "openbci_eeg_id43"
    PATH_RUN = HOST_NAME  # for saving data later on 
    CH_TYPE = "ecog"  # pretend ECoG signals, s.t. define_M1 works 

    # read settings
    settings_wrapper = nm_settings.SettingsWrapper('settings_LSL.json')

    # (if available) add coordinates to settings here
    '''
    if settings_wrapper.settings["methods"]["project_cortex"] is True or \
            settings_wrapper.settings["methods"]["project_subcortex"] is True:
        settings_wrapper.add_coord(raw_arr.copy())  # if not copy ch_names is being set
        projection_ = projection.Projection(settings_wrapper.settings)
    else:
        projection_ = None
    '''

    # test settings
    test_settings(settings_wrapper.settings, verbose=True)

    line_noise = 50

    with LSLClient(host=HOST_NAME, wait_max=10) as client:

        print("client initialized")
        client_info = client.get_measurement_info()

        ch_names = client_info["ch_names"]
        ch_types = ["ECOG" for i in range(len(ch_names))]
        sfreq = int(client_info['sfreq'])

        # test if client receives data
        # for i in range(19):
        #    print(client.get_data_as_epoch(n_samples=sfreq)._data.shape)

        # define M1
        settings_wrapper.set_M1(m1_path=None, ch_names=ch_names, ch_types=ch_types)
        settings_wrapper.set_fs_line_noise(sfreq, line_noise) 
    
         # initialize rereferencing
        if settings_wrapper.settings["methods"]["re_referencing"] is True:
            rereference_ = rereference.RT_rereference(settings_wrapper.df_M1)
        else:
            rereference_ = None
        
        # define resampler for faster feature estimation
        if settings_wrapper.settings["methods"]["raw_resampling"] is True:
            resample_ = resample.Resample(settings_wrapper.settings)
        else:
            resample_ = None

        # initialize feature class from settings
        features_ = features.Features(settings_wrapper.settings)
    
        # initialize run object
        run_analysis_ = run_analysis.Run(features_, settings_wrapper.settings,
                                     rereference_, projection=None, resample=resample_, verbose=True)
        
        # this loop is now asynchron; replace later with thread that runs at given time
        # in a cue run_analysis.run could then estimate the data batches
        # run_analyis.run in fact expects every settings["sampling_rate_features"] a new data batch 

        counter_samples = 0
        while counter_samples < 100:
            ieeg_batch = np.squeeze(client.get_data_as_epoch(n_samples=sfreq))
            run_analysis_.run(ieeg_batch)
            counter_samples += 1
        
        IO.save_features_and_settings(df_=run_analysis_.feature_arr, run_analysis_=run_analysis_,
                                  folder_name=os.path.basename(PATH_RUN)[:-5],
                                  settings_wrapper=settings_wrapper)



#####################

'''
        ch_names = settings_wrapper.df_M1['name'].to_numpy()
        feature_idx, = np.where(np.logical_and(np.array((settings_wrapper.df_M1["used"] == 1)),
                                           np.array((settings_wrapper.df_M1["target"] == 0))))
        used_chs = ch_names[feature_idx].tolist()

        # initialize feature class from settings
        features_ = features.Features(s=settings, fs=sfreq, line_noise=line_noise,
                                      channels=used_chs)

        df_ = run_analysis.run(gen=None, features=features_, settings=settings,
                               ref_here=ref_here, used=feature_idx, resample_=None, 
                               client=client)

        # create out folder if doesn't exist
        folder_name = os.path.basename(PATH_RUN)[:-5]
        if not os.path.exists(os.path.join(settings["out_path"], folder_name)):
            print("create output folder "+str(folder_name))
            os.makedirs(os.path.join(settings["out_path"], folder_name))

        df_.to_csv(os.path.join(settings["out_path"], folder_name,
                                folder_name+"_FEATURES.csv"))

        # save used settings and coordinates to settings as well
        # this becomes necessary since MNE RawArry Info cannot be saved as json
        settings["sfreq"] = sfreq

        with open(os.path.join(settings["out_path"], folder_name,
                               folder_name+'_SETTINGS.json'), 'w') as f:
            json.dump(settings, f)

        # save df_M1 as csv
        df_M1.to_csv(os.path.join(settings["out_path"], folder_name,
                                  folder_name+"_DF_M1.csv"))
'''