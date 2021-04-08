import sys 
import os
from bids import BIDSLayout
from itertools import product
from mne_realtime import LSLClient
import json
import numpy as np

sys.path.append(
    r'C:\Users\ICN_admin\Documents\py_neuromodulation\pyneuromodulation')
from settings import test_settings
import define_M1
import rereference
import features
import run_analysis
import generator_LSL

if __name__ == "__main__":

    # M1 needs to be defined 

    HOST_NAME = "openbci_eeg_id43"
    PATH_RUN = HOST_NAME  # for saving data later on 
    CH_TYPE = "ecog"  # pretend ECoG signals, s.t. define_M1 works 
    # read settings
    with open('settings.json', encoding='utf-8') as json_file:
        settings = json.load(json_file)

    # test settings
    test_settings(settings, verbose=True)

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
        df_M1 = define_M1.set_M1(ch_names, ch_types)

        ref_here = rereference.RT_rereference(df_M1, split_data=False)

        ch_names = df_M1['name'].to_numpy()
        feature_idx, = np.where(np.logical_and(np.array((df_M1["used"] == 1)),
                                           np.array((df_M1["target"] == 0))))
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