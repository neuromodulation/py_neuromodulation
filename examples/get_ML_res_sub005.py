import sys 
import os
import numpy as np
from pathlib import Path
from scipy import stats
from matplotlib import pyplot as plt
import _pickle as cPickle
from scipy import io

PATH_PYNEUROMODULATION = Path(__file__).absolute().parent.parent
PATH_PLOT = os.path.join(Path(__file__).absolute().parent.parent, 'plots')
sys.path.append(os.path.join(PATH_PYNEUROMODULATION, 'pyneuromodulation'))
import nm_reader

FEATURE_PATH = r'C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\sub005'
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


def read_ind_channel_results(feature_str, performance_dict):
    DEFAULT_PERFORMANCE = 0.5

    PATH_ML_ = os.path.join(FEATURE_PATH, feature_str, feature_str + "_XGB_ML_RES.p")
    
    # read ML results
    with open(PATH_ML_, 'rb') as input: 
        ML_res = cPickle.load(input)
    # read here the coordinates and save them in meta file / plot them
    # all results are for now contralateral

    performance_dict[feature_str] = {}  # subject

    # channels 
    # use here only ECoG for now
    ch_to_use = list(np.array(ML_res.settings["ch_names"])[np.where(np.array(ML_res.settings["ch_types"]) == 'ecog')[0]])
    for ch in ch_to_use:

        performance_dict[feature_str][ch] = {}  # should be 7 for Berlin

        ML_res.settings["sess_right"] = True  # HACK, wasn't written 

        if ML_res.settings["sess_right"] is True:
            cortex_name = "cortex_right"
        else:
            cortex_name = "cortex_left"
        
        #idx_ = np.where(ch == np.array(ML_res.settings["coord"][cortex_name]["ch_names"]))[0][0]
        #coords = ML_res.settings["coord"][cortex_name]["positions"][idx_]
        #performance_dict[feature_str[4:10]][ch]["coord"] = coords
        performance_dict[feature_str][ch]["performance"] = np.mean(ML_res.ch_ind_pr[ch]["score_test"])

    return performance_dict

if __name__ == "__main__":
    
    nm_reader = nm_reader.NM_Reader(FEATURE_PATH)
    feature_list = nm_reader.get_feature_list()

    # run ML for this run
    performance_dict = {}
    for feature_str in feature_list:
        # read ML results across patients
        performance_dict = read_ind_channel_results(feature_str, performance_dict)
    
    