import sys 
import os
import numpy as np
from pathlib import Path
from scipy import stats
from matplotlib import pyplot as plt

PATH_PYNEUROMODULATION = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation'))

import nm_reader as NM_reader

if __name__ == "__main__":

    FEATURE_PATH = r'C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\sub005'

    nm_reader = NM_reader.NM_Reader(FEATURE_PATH)

    feature_list = nm_reader.get_feature_list()

    for feature_file in feature_list:

        settings = nm_reader.read_settings(feature_file)

        # read run_analysis
        run_anylyzer = nm_reader.read_run_analyzer()

        # optionally read ML estimations
        #ML_est = nm_reader.read_ML_estimations()

        # plot cortical projection
        PATH_PLOT = os.path.join(PATH_PYNEUROMODULATION, 'plots')
        nm_reader.read_plot_modules(PATH_PLOT)

        _ = nm_reader.read_M1(feature_file)
        _ = nm_reader.read_features(feature_file)

        

        if feature_file == 'sub-005_ses-EphysMedOff01_task-SelfpacedRotationL_acq-StimOff_run-01_ieeg':
            label_name = "ANALOG_L_ROTA_CH"
            dat_label = nm_reader.read_label(label_name)
            nm_reader.label = np.nan_to_num(np.array(dat_label)) > 4*10**(-7)
            ch_name = 'ECOG_R_1_SMC_AT-avgref'
        elif feature_file == 'sub-005_ses-EphysMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg':
            label_name = "ANALOG_R_ROTA_CH"
            dat_label = nm_reader.read_label(label_name)
            nm_reader.label = np.nan_to_num(np.array(-dat_label)) > 0
        elif feature_file == 'sub-005_ses-EphysMedOn01_task-SelfpacedRotationL_acq-StimOff_run-01_ieeg':
            label_name = "ANALOG_L_ROTA_CH"
            dat_label = nm_reader.read_label(label_name)
            nm_reader.label = np.nan_to_num(np.array(-2*10**(-7)+dat_label)) > 0
        elif feature_file == 'sub-005_ses-EphysMedOn01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg':
            label_name = "ANALOG_R_ROTA_CH"
            dat_label = nm_reader.read_label(label_name)
            nm_reader.label = np.nan_to_num(np.array(-dat_label)) > 0

        plt.figure()
        plt.plot(nm_reader.label)
        plt.show()


        

        # Fist case: filter for bandpass activity features only
        dat_ch = nm_reader.read_channel_data(ch_name, read_bp_activity_only=True)

        # estimating epochs, with shape (epochs,samples,channels,features)
        X_epoch, y_epoch = nm_reader.get_epochs_ch(epoch_len=4,
                                                    sfreq=settings["sampling_rate_features"],
                                                    threshold=0.1)  # epoch_len in seconds
        print("plotting feature covariance matrix")
        nm_reader.plot_corr_matrix(feature_file, feature_str_add="bandpass")
        print("plotting feature target averaged")
        nm_reader.plot_epochs_avg(feature_file, feature_str_add="bandpass")
