import sys 
import os
import numpy as np 
from scipy import stats

os.chdir(os.path.join(os.pardir,'pyneuromodulation'))
sys.path.append(os.path.join(os.pardir,'pyneuromodulation'))

import nm_reader
import nm_decode
import multiprocessing
from sklearn import linear_model
from sklearn import metrics 
from sklearn import model_selection
import xgboost


PATH_PYNEUROMODULATION = os.getcwd()
PATH_PLOT = os.path.join(os.pardir, 'plots')

def write_proj_and_avg_features(feature_file):
    """Plot projected grid. Plot movement aligned features

    Parameters
    ----------
    feature_file : string
        path of feature folder
    """
    settings =  nm_reader.read_settings(feature_file)

    run_anylyzer = nm_reader.read_run_analyzer()

    nm_reader.read_plot_modules(PATH_PLOT)
    nm_reader.plot_cortical_projection()

    # optionally read ML estimations
    #ML_est = nm_reader.read_ML_estimations()

    df_M1 = nm_reader.read_M1(feature_file)
    _ = nm_reader.read_features(feature_file)

    # get first ECoG chnnel names 
    ch_name = [i for i in settings["ch_names"] if "ECOG" in i][0]
    dat_ch = nm_reader.read_channel_data(ch_name, read_bp_activity_only=True)

    # get first target, try to get contralateral target 
    label_names = list(df_M1["name"][df_M1["target"] == 1])

    if settings["sess_right"] is True:
        label_name = "MOV_LEFT_CLEAN"
    else:
        label_name = "MOV_RIGHT_CLEAN"
    if label_name not in label_names:
        # just pick first label in this case
        label_name = label_names[0]
    print("label_name: "+str(label_name))
    dat_label = nm_reader.read_label(label_name)
    nm_reader.label = stats.zscore(-dat_label)

    # the threshold parameter is important here
    # many labels have a constant offset, which resembles then baseline
    X_epoch, y_epoch = nm_reader.get_epochs_ch(epoch_len=4,
                                                sfreq=settings["sampling_rate_features"],
                                                threshold=0.5)

    #nm_reader.plot_corr_matrix(feature_file)
    nm_reader.plot_epochs_avg(feature_file)

def run_ML_single_channel(feature_str):
    model = linear_model.LinearRegression()
    decoder = nm_decode.Decoder(feature_path=FEATURE_PATH,
                                feature_file=feature_str,
                                model=model,
                                eval_method=metrics.r2_score,
                                cv_method=model_selection.KFold(n_splits=3, shuffle=False),
                                threshold_score=True
                                )
    
    # run estimations for channels individually 
    decoder.set_data_ind_channels()
    decoder.run_CV_ind_channels()

    decoder.save("LM")

def run_ML_single_channel_XGB(feature_str):
    model = xgboost.XGBClassifier()
    decoder = nm_decode.Decoder(feature_path=FEATURE_PATH,
                                feature_file=feature_str,
                                model=model,
                                eval_method=metrics.accuracy_score,
                                cv_method=model_selection.KFold(n_splits=3, shuffle=False),
                                threshold_score=True
                                )
    
    # run estimations for channels individually 
    decoder.set_data_ind_channels()
    decoder.run_CV_ind_channels()

    decoder.save("XGB")

FEATURE_PATH = r'C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\Pittsburgh'
if __name__ == "__main__":

    
    nm_reader = nm_reader.NM_Reader(FEATURE_PATH)
    feature_list = nm_reader.get_feature_list()

    # plot projection 
    '''
    for feature_file in feature_list:
        write_proj_and_avg_features(feature_file)
    or when being sure: pool
    #pool = multiprocessing.Pool(processes=8)
    #pool.map(write_proj_and_avg_features, feature_list)
    '''

    # run LM estimation for
    #pool = multiprocessing.Pool(processes=30)
    #pool.map(run_ML_single_channel, feature_list)
    for feature_str in feature_list:
        run_ML_single_channel_XGB(feature_str)
