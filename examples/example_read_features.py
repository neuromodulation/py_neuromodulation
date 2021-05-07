import sys 
import os
import numpy as np
from pathlib import Path
from scipy import stats

PATH_PYNEUROMODULATION = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation'))

import nm_reader as NM_reader

def run_example_read_features():

    FEATURE_PATH = os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation',
                                'tests', 'data', 'derivatives')
    nm_reader = NM_reader.NM_Reader(FEATURE_PATH)

    feature_list = nm_reader.get_feature_list()
    feature_file = feature_list[0]

    settings = nm_reader.read_settings(feature_file)

    # read run_analysis
    run_anylyzer = nm_reader.read_run_analyzer()

    # optionally read ML estimations
    ML_est = nm_reader.read_ML_estimations()

    # plot cortical projection
    print("plotting cortical projections")
    PATH_PLOT = os.path.join(PATH_PYNEUROMODULATION, 'plots')
    nm_reader.read_plot_modules(PATH_PLOT)
    nm_reader.plot_cortical_projection()

    _ = nm_reader.read_M1(feature_file)
    _ = nm_reader.read_features(feature_file)

    label_name = "ANALOG_ROT_R_1"
    dat_label = nm_reader.read_label(label_name)
    nm_reader.label = stats.zscore(dat_label)*-1  # the label is flipped in the example

    ch_name = "ECOG_AT_SM_L_6"

    # Fist case: filter for bandpass activity features only
    dat_ch = nm_reader.read_channel_data(ch_name, read_bp_activity_only=True)

    # estimating epochs, with shape (epochs,samples,channels,features)
    X_epoch, y_epoch = nm_reader.get_epochs_ch(epoch_len=4,
                                               sfreq=settings["sampling_rate_features"],
                                               threshold=0.1)
    print("plotting feature covariance matrix")
    nm_reader.plot_corr_matrix(feature_file, feature_str_add="bandpass")
    print("plotting feature target averaged")
    nm_reader.plot_epochs_avg(feature_file, feature_str_add="bandpass")


    # Second case: filter for sharpwave prominence features only
    dat_ch = nm_reader.read_channel_data(ch_name, read_sharpwave_prominence_only=True)

    # estimating epochs, with shape (epochs,samples,channels,features)
    X_epoch, y_epoch = nm_reader.get_epochs_ch(epoch_len=4,
                                               sfreq=settings["sampling_rate_features"],
                                               threshold=0.1)
    print("plotting feature covariance matrix")
    nm_reader.plot_corr_matrix(feature_file, feature_str_add="sharpwaveprominence")
    print("plotting feature target averaged")
    nm_reader.plot_epochs_avg(feature_file, feature_str_add="sharpwaveprominence")

if __name__ == "__main__":

    run_example_read_features()