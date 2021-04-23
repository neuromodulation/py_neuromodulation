import sys 
import os
import numpy as np

os.chdir(os.path.join(os.pardir,'pyneuromodulation'))
sys.path.append(os.path.join(os.pardir,'pyneuromodulation'))

import nm_reader

if __name__ == "__main__":

    PATH_PYNEUROMODULATION = os.getcwd()

    FEATURE_PATH = os.path.join(PATH_PYNEUROMODULATION, 'tests', 'data', 'derivatives')
    nm_reader = nm_reader.NM_Reader(FEATURE_PATH)

    feature_list = nm_reader.get_feature_list()
    feature_file = feature_list[0]

    settings = nm_reader.read_settings(feature_file)

    # read run_analysis
    run_anylyzer = nm_reader.read_run_analyzer()

    # plot cortical projection
    print("plotting cortical projections")
    PATH_PLOT = os.path.join(os.pardir, 'plots')
    nm_reader.read_plot_modules(PATH_PLOT)
    nm_reader.plot_cortical_projection()

    # optionally read ML estimations
    ML_est = nm_reader.read_ML_estimations()

    _ = nm_reader.read_M1(feature_file)
    _ = nm_reader.read_features(feature_file)

    ch_name = "ECOG_AT_SM_L_6"
    dat_ch = nm_reader.read_channel_data(ch_name)

    label_name = "ANALOG_ROT_R_1"
    dat_label = nm_reader.read_label(label_name)

    # estimating epochs, with shape (epochs,samples,channels,features)
    X_epoch, y_epoch = nm_reader.get_epochs_ch(epoch_len=2,
                                               sfreq=settings["sampling_rate_features"],
                                               threshold=0.1)

    print("plotting feature covariance matrix")
    nm_reader.plot_corr_matrix(feature_file)
    print("plotting feature target averaged")
    nm_reader.plot_epochs_avg(feature_file)
