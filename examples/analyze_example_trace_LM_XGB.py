import sys 
import os
import matplotlib
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
import _pickle as cPickle
from scipy import io
from matplotlib import pyplot as plt
import matplotlib


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

def run_ML_LM(feature_str):
    model = linear_model.LogisticRegression()
    decoder = nm_decode.Decoder(feature_path=FEATURE_PATH,
                                feature_file=feature_str,
                                model=model,
                                eval_method=metrics.balanced_accuracy_score,
                                cv_method=model_selection.KFold(n_splits=3, shuffle=False),
                                threshold_score=True
                                )
    
     # run estimations for channels and grid points individually  
     # currently the MLp file get's saved only, and overwrited previous files 
    decoder.set_data_ind_channels()
    decoder.set_data_grid_points()

    decoder.run_CV_ind_channels(XGB=False)
    decoder.run_CV_grid_points(XGB=False)
    decoder.save("LM")

def run_ML_XGB(feature_str):
    model = xgboost.XGBClassifier()
    decoder = nm_decode.Decoder(feature_path=FEATURE_PATH,
                                feature_file=feature_str,
                                model=model,
                                eval_method=metrics.balanced_accuracy_score,
                                cv_method=model_selection.KFold(n_splits=3, shuffle=False),
                                threshold_score=True
                                )
    
    # run estimations for channels and grid points individually 
    decoder.set_data_ind_channels()
    decoder.set_data_grid_points()

    decoder.run_CV_ind_channels(XGB=True)
    decoder.run_CV_grid_points(XGB=True)
    decoder.save("XGB")


def read_ind_channel_results(feature_str, performance_dict):
    DEFAULT_PERFORMANCE = 0.5


    PATH_ML_ = os.path.join(FEATURE_PATH, feature_str, feature_str + "_XGB_ML_RES.p")
    
    # read ML results
    with open(PATH_ML_, 'rb') as input: 
        ML_res = cPickle.load(input)
    # read here the coordinates and save them in meta file / plot them
    # all results are for now contralateral

    performance_dict[feature_str[4:10]] = {}  # subject

    # channels 
    # use here only ECoG for now
    ch_to_use = list(np.array(ML_res.settings["ch_names"])[np.where(np.array(ML_res.settings["ch_types"]) == 'ecog')[0]])
    for ch in ch_to_use:

        performance_dict[feature_str[4:10]][ch] = {}  # should be 7 for Berlin

        if ML_res.settings["sess_right"] is True:
            cortex_name = "cortex_right"
        else:
            cortex_name = "cortex_left"
        
        idx_ = np.where(ch == np.array(ML_res.settings["coord"][cortex_name]["ch_names"]))[0][0]
        coords = ML_res.settings["coord"][cortex_name]["positions"][idx_]
        performance_dict[feature_str[4:10]][ch]["coord"] = coords
        performance_dict[feature_str[4:10]][ch]["performance"] = np.mean(ML_res.ch_ind_pr[ch]["score_test"])

    # read now also grid point results
    for grid_point in range(len(ML_res.settings["grid_cortex"])):
        performance_dict["grid_"+str(grid_point)] = {}
        performance_dict["grid_"+str(grid_point)]["coord"] = ML_res.settings["grid_cortex"][grid_point]
        if grid_point in ML_res.active_gridpoints:
            performance_dict["grid_"+str(grid_point)]["performance"] = np.mean(ML_res.gridpoint_ind_pr[grid_point]["score_test"])
        else:
            performance_dict["grid_"+str(grid_point)]["performance"] = DEFAULT_PERFORMANCE
    return performance_dict


def plot_cohort_performances(performance_dict=None):

    if performance_dict is None:
        with open(r'C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\META\Beijing_out.p', 'rb') as input: 
            performance_dict = cPickle.load(input)
    
    subjects_ = [sub for sub in list(performance_dict.keys()) if "grid" not in sub]
    
    per_ = []
    coords_ = []
    for sub in subjects_:
        ch_ = performance_dict[sub].keys()
        for ch in ch_:
            coords_.append(performance_dict[sub][ch]["coord"])
            per_.append(performance_dict[sub][ch]["performance"])
    
    ecog_strip = np.vstack(coords_).T
    ecog_strip_per = per_
    fig, axes = plt.subplots(1,1, facecolor=(1,1,1), \
                                figsize=(14,9))#, dpi=300)
    axes.scatter(x_ecog, y_ecog, c="gray", s=0.001)
    axes.axes.set_aspect('equal', anchor='C')

    pos_elec = axes.scatter(ecog_strip[0,:],
                            ecog_strip[1,:], c=ecog_strip_per, 
                            s=50, alpha=0.8, cmap="viridis", marker="x")

    # in case the grid is plotted
    #pos_ecog = axes.scatter(cortex_grid[0,:],
    #                            cortex_grid[1,:], c=grid_color, 
    #                            s=30, alpha=0.8, cmap="viridis")

    plt.axis('off')
    pos_elec.set_clim(0.5,0.8)
    fig.colorbar(pos_elec)

    plt.savefig(r'C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\META\PBeijing_Cohort_ch.png', bbox_inches = "tight")
    print("saved Figure")

# needs to be accesible for all functions
FEATURE_PATH = r'C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\Beijing'
faces = io.loadmat(os.path.join(PATH_PLOT, 'faces.mat'))
vertices = io.loadmat(os.path.join(PATH_PLOT, 'Vertices.mat'))
grid = io.loadmat(os.path.join(PATH_PLOT, 'grid.mat'))['grid']
stn_surf = io.loadmat(os.path.join(PATH_PLOT, 'STN_surf.mat'))
x_ver = stn_surf['vertices'][::2,0]
y_ver = stn_surf['vertices'][::2,1]
x_ecog = vertices['Vertices'][::1,0]
y_ecog = vertices['Vertices'][::1,1]
z_ecog = vertices['Vertices'][::1,2]
x_stn = stn_surf['vertices'][::1,0]
y_stn = stn_surf['vertices'][::1,1]
z_stn = stn_surf['vertices'][::1,2]

if __name__ == "__main__":


    

    # call in the upper function the grid points and ind channel results
    # then save! 
    # and repeat for LM
    
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

    # run ML analysis
    '''
    for feature_str in feature_list:
        run_ML_XGB(feature_str)
    '''
    # run LM estimation multiprocessing
    #pool = multiprocessing.Pool(processes=30)
    #pool.map(run_ML_single_channel, feature_list)

    performance_dict = {}
    for feature_str in feature_list:
        # read ML results across patients
        performance_dict = read_ind_channel_results(feature_str, performance_dict)
    
    # now save as meta analysis data 
    # save a dict with concatenated arrays for channels 
    with open(r'C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\META\Beijing_out.p', 'wb') as output:  
            cPickle.dump(performance_dict, output)
    plot_cohort_performances(performance_dict=performance_dict)