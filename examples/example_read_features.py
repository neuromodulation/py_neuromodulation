import sys 
import os
import numpy as np

sys.path.append(
    r'C:\Users\ICN_admin\Documents\py_neuromodulation\pyneuromodulation')
import nm_reader

if __name__ == "__main__":

    PATH_PYNEUROMODULATION = r"C:\Users\ICN_admin\Documents\py_neuromodulation"
    FEATURE_PATH = os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation', 'tests', 'data', 'derivatives')
    nm_reader = nm_reader.NM_Reader(FEATURE_PATH)

    feature_list = nm_reader.get_feature_list()
    feature_file = feature_list[0]

    settings = nm_reader.read_settings(feature_file)

    # read run_analysis
    run_anylyzer = nm_reader.read_run_analyzer()

    # optionally read ML estimations
    ML_est = nm_reader.read_ML_estimations()

    _ = nm_reader.read_M1(feature_file)
    _ = nm_reader.read_file(feature_file)

    ch_name = "ECOG_AT_SM_L_6"
    dat_ch = nm_reader.read_channel_data(ch_name)

    label_name = "ANALOG_ROT_R_1"
    dat_label = nm_reader.read_label(label_name)

    X_epoch, y_epoch = nm_reader.get_epochs_ch(epoch_len=1,
                                               sfreq=settings["sampling_rate_features"],
                                               threshold=0.1)

    nm_reader.plot_corr_matrix(feature_file)
    nm_reader.plot_epochs_avg(feature_file)
