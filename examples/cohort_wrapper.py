import sys
import os
import numpy as np
from pathlib import Path
from scipy import stats
import multiprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn.base import clone
from sklearn import model_selection
from sklearn.utils import class_weight
import xgboost
import _pickle as cPickle
from scipy import io
from matplotlib import pyplot as plt
import matplotlib
import bids
from bids import BIDSLayout

from pyneuromodulation import nm_reader as NM_reader
from pyneuromodulation import nm_decode, nm_start_BIDS, nm_settings, nm_analysis


def multiprocess_pipeline_run_wrapper(PATH_RUN, ML_model_name="LM",
                                      model=linear_model.LogisticRegression(class_weight="balanced")):

    # This function assumes that the nm_settings.json are correct! In other files, e.g. nm_analysis the settings
    # are also read again
    if type(PATH_RUN) is bids.layout.models.BIDSFile:
        PATH_RUN = PATH_RUN.path

    settings_wrapper = nm_settings.SettingsWrapper(settings_path=(r"C:\Users\ICN_admin\Documents\py_neuromodulation\
                                                   pyneuromodulation\nm_settings.json"))

    nm_BIDS = nm_start_BIDS.NM_BIDS(PATH_RUN, ECOG_ONLY=True)
    nm_BIDS.run_bids()

    feature_path = settings_wrapper.settings["out_path"]
    feature_file = os.path.basename(PATH_RUN)[:-5]  # cut off ".vhdr"

    feature_wrapper = nm_analysis.FeatureReadWrapper(feature_path, feature_file,
                                                     plt_cort_projection=True)
    feature_wrapper.read_plotting_modules()

    feature_wrapper.plot_features()
    feature_wrapper.run_ML_model(model=model, output_name=ML_model_name, TRAIN_VAL_SPLIT=False)
    feature_wrapper.run_ML_model()
    performance_dict = {}

    # subject_name is different across cohorts
    subject_name = feature_wrapper.feature_file[4:10]
    performance_dict = feature_wrapper.read_ind_channel_results(performance_dict,
                                                                subject_name)

    feature_wrapper.plot_subject_grid_ch_performance(subject_name,
                                                     performance_dict=performance_dict,
                                                     plt_grid=True, output_name=ML_model_name)


def run_cohort(cohort="Pittsburgh"):
    """example of instantiating multiprocessing pool for running multiple run files

    Parameters
    ----------
    cohort : str, optional
        name of available cohort, by default "Pittsburgh"
    """
    if cohort == "Berlin":
        PATH_BIDS = "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\Data\\Berlin_VoluntaryMovement"
        layout = BIDSLayout(PATH_BIDS)
        run_files = layout.get(extension='.vhdr')
    elif cohort == "Beijing":
        PATH_BIDS = "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\Data\\Beijing"
        layout = BIDSLayout(PATH_BIDS)
        subjects = layout.get_subjects()
        run_files = []
        for sub in subjects:
            if sub != "FOG013":
                run_files.append(layout.get(subject=sub, task='ButtonPress', extension='.vhdr')[0])
    elif cohort == "Pittsburgh":
        PATH_BIDS = "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\Data\\Pittsburgh"
        layout = BIDSLayout(PATH_BIDS)
        run_files = layout.get(extension='.vhdr')

    # check OPTIONALLY which 'runs' where not estimated through yet, and run those
    run_files_left = []
    directory = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\try_1708\Berlin"
    folders_path = [x[0] for x in os.walk(directory)]
    folders = [os.path.basename(x) for x in folders_path[1:]]
    for run_file in run_files:
        feature_file = os.path.basename(run_file.path)[:-5]
        if feature_file not in folders:
            run_files_left.append(run_file)

    # multiprocess_pipeline_run_wrapper(run_files[0])
    pool = multiprocessing.Pool(processes=55)  # most on Ryzen CPU 2990WX is 63
    pool.map(multiprocess_pipeline_run_wrapper, run_files)


def read_cohort_results(feature_path, cohort, ML_model_name="LM"):
    """Read for a given path (of potentially multiple estimated runs) performance results

    Parameters
    ----------
    feature_path : string
        path where estimated runs are saved
    cohort : string
        used for saving output npy dictionary
    ML_model_name : string
        model name, by default "LM"
    """
    folders_path = [x[0] for x in os.walk(feature_path)]
    feature_paths = [os.path.basename(x) for x in folders_path[1:]]
    performance_dict = {}

    for feature_file in feature_paths:
        feature_wrapper = nm_analysis.FeatureReadWrapper(feature_path, feature_file,
                                                         plt_cort_projection=False, read_features=False)
        subject_name = feature_file[feature_file.find("sub-"):feature_file.find("_ses")]
        # cut here s.t. the subject is the whole recording
        subject_name = feature_file[:-5]
        performance_dict = feature_wrapper.read_ind_channel_results(performance_dict,
                                                                    subject_name, ML_model_name=ML_model_name)
    np.save(ML_model_name+'_cohort_'+cohort+'.npy', performance_dict)


def cohort_wrapper_read_cohort(feature_path="C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\write_out\\try_1708",
                               ML_model_name="LM"):
    """Read results for multiple cohorts

    Parameters
    ----------
    feature_path : str, optional
        path to feature path, by default "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\write_out\\try_0408"
    model : string
        model_name name, by default "LM"
    """
    cohorts = ["Pittsburgh", "Beijing", "Berlin"]
    for cohort in cohorts:
        read_cohort_results(os.path.join(feature_path, cohort), cohort, ML_model_name)


def read_all_grid_points(grid_point_all, feature_path, feature_file, cohort):
    """Save for a given feature path all used grid point data. Necessary to run across patient and cohort analysis.

    Parameters
    ----------
    grid_point_all : dictionary
        dictionary with data, label, label_name and feature_names for each grid point
    feature_path : string
        path to feature files
    feature_file : string
        feature file
    cohort : string
        used for indecing of grid_point_all

    Returns
    -------
    dictionary
        grid_point_all
    """
    feature_wrapper = nm_analysis.FeatureReadWrapper(feature_path, feature_file,
                                                     plt_cort_projection=False)
    decoder = nm_decode.Decoder(feature_path=feature_path,
                                feature_file=feature_file)
    decoder.label = feature_wrapper.nm_reader.label
    decoder.target_ch = feature_wrapper.label_name
    decoder.set_data_grid_points()
    subject_name = feature_file[feature_file.find("sub-")+4:feature_file.find("_ses")]
    sess_name = feature_file[feature_file.find("ses-")+4:feature_file.find("_task")]
    task_name = feature_file[feature_file.find("task-")+5:feature_file.find("_run")]
    run_number = feature_file[feature_file.find("run-")+4:feature_file.find("_ieeg")]

    for grid_point in list(decoder.grid_point_ind_data.keys()):
        if grid_point not in grid_point_all:
            grid_point_all[grid_point] = {}
        if cohort not in grid_point_all[grid_point]:
            grid_point_all[grid_point][cohort] = {}
        if subject_name not in grid_point_all[grid_point][cohort]:
            grid_point_all[grid_point][cohort][subject_name] = {}
        grid_point_all[grid_point][cohort][subject_name][feature_file] = {}

        grid_point_all[grid_point][cohort][subject_name][feature_file]["data"] = decoder.grid_point_ind_data[grid_point]
        grid_point_all[grid_point][cohort][subject_name][feature_file]["feature_names"] = decoder.feature_names
        grid_point_all[grid_point][cohort][subject_name][feature_file]["label"] = decoder.label
        grid_point_all[grid_point][cohort][subject_name][feature_file]["label_name"] = decoder.target_ch

        # check laterality
        lat = "CON"  # Beijing is always contralateral
        # Pittsburgh Subjects
        if ("LEFT" in decoder.target_ch and "LEFT" in decoder.run_analysis.features.ch_names[0]) or \
           ("RIGHT" in decoder.target_ch and "RIGHT" in decoder.run_analysis.features.ch_names[0]):
            lat = "IPS"

        # Berlin subjects
        if ("_L_" in decoder.run_analysis.features.ch_names[0] and task_name == "SelfpacedRotationL") or \
           ("_R_" in decoder.run_analysis.features.ch_names[0] and task_name == "SelfpacedRotationR"):
            lat = "IPS"
        grid_point_all[grid_point][cohort][subject_name][feature_file]["lat"] = lat
    return grid_point_all


def cohort_wrapper_read_all_grid_points(feature_path_cohorts=(r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\
                                                              write_out\try_0408")):
    cohorts = ["Pittsburgh", "Beijing", "Berlin"]
    grid_point_all = {}
    for cohort in cohorts:
        print("COHORT: "+cohort)
        feature_path = os.path.join(feature_path_cohorts, cohort)
        nm_reader = NM_reader.NM_Reader(feature_path)
        feature_list = nm_reader.get_feature_list()
        for feature_file in feature_list:
            print(feature_file)
            grid_point_all = read_all_grid_points(grid_point_all, feature_path, feature_file, cohort)

    np.save('grid_point_all.npy', grid_point_all)


def run_cohort_leave_one_patient_out_CV(feature_path=(r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\
                                                      try_0408\Beijing"),
                                        model_base=linear_model.LogisticRegression(class_weight="balanced"),
                                        ML_model_name="LM"):

    grid_point_all = np.load('grid_point_all.npy', allow_pickle='TRUE').item()
    performance_leave_one_patient_out = {}

    for cohort in ["Pittsburgh", "Beijing", "Berlin"]:
        print('cohort: '+str(cohort))
        performance_leave_one_patient_out[cohort] = {}

        for grid_point in list(grid_point_all.keys()):
            print('grid point: '+str(grid_point))
            if cohort not in grid_point_all[grid_point]:
                continue
            if len(list(grid_point_all[grid_point][cohort].keys())) <= 1:
                continue  # cannot do leave one out prediction with a single subject
            performance_leave_one_patient_out[cohort][grid_point] = {}

            for subject_test in list(grid_point_all[grid_point][cohort].keys()):
                X_test = []
                y_test = []
                for run in list(grid_point_all[grid_point][cohort][subject_test].keys()):
                    if grid_point_all[grid_point][cohort][subject_test][run]["lat"] != "CON":
                        continue
                    X_test.append(grid_point_all[grid_point][cohort][subject_test][run]["data"])
                    y_test.append(grid_point_all[grid_point][cohort][subject_test][run]["label"])
                if len(X_test) > 1:
                    X_test = np.concatenate(X_test, axis=0)
                    y_test = np.concatenate(y_test, axis=0)
                else:
                    X_test = X_test[0]
                    y_test = y_test[0]
                X_train = []
                y_train = []
                for subject_train in list(grid_point_all[grid_point][cohort].keys()):
                    if subject_test == subject_train:
                        continue
                    for run in list(grid_point_all[grid_point][cohort][subject_train].keys()):
                        if grid_point_all[grid_point][cohort][subject_train][run]["lat"] != "CON":
                            continue
                        X_train.append(grid_point_all[grid_point][cohort][subject_train][run]["data"])
                        y_train.append(grid_point_all[grid_point][cohort][subject_train][run]["label"])
                if len(X_train) > 1:
                    X_train = np.concatenate(X_train, axis=0)
                    y_train = np.concatenate(y_train, axis=0)
                else:
                    X_train = X_train[0]
                    y_train = y_train[0]

                model = clone(model_base)
                # run here ML estimation
                if ML_model_name == "XGB":
                    X_train, X_val, y_train, y_val = \
                        model_selection.train_test_split(
                            X_train, y_train, train_size=0.7, shuffle=False)
                    classes_weights = class_weight.compute_sample_weight(
                        class_weight='balanced', y=y_train)

                    model.fit(
                        X_train, y_train, eval_set=[(X_val, y_val)],
                        early_stopping_rounds=7, sample_weight=classes_weights,
                        verbose=False)
                else:
                    # LM
                    model.fit(X_train, y_train)

                y_tr_pr = model.predict(X_train)
                y_te_pr = model.predict(X_test)
                performance_leave_one_patient_out[cohort][grid_point][subject_test] = {}
                performance_leave_one_patient_out[cohort][grid_point][subject_test]["y_test"] = y_test
                performance_leave_one_patient_out[cohort][grid_point][subject_test]["y_test_pr"] = y_te_pr
                performance_leave_one_patient_out[cohort][grid_point][subject_test]["y_train"] = y_train
                performance_leave_one_patient_out[cohort][grid_point][subject_test]["y_train_pr"] = y_tr_pr
                performance_leave_one_patient_out[cohort][grid_point][subject_test]["performance_test"] = \
                    metrics.balanced_accuracy_score(y_test, y_te_pr)
                performance_leave_one_patient_out[cohort][grid_point][subject_test]["performance_train"] = \
                    metrics.balanced_accuracy_score(y_train, y_tr_pr)

    # add the cortex grid for plotting
    nm_reader = NM_reader.NM_Reader(feature_path)
    feature_file = nm_reader.get_feature_list()[0]
    grid_cortex = np.array(nm_reader.read_settings(feature_file)["grid_cortex"])
    performance_leave_one_patient_out["grid_cortex"] = grid_cortex
    np.save(ML_model_name+'_performance_leave_one_patient_out.npy', performance_leave_one_patient_out)
    return performance_leave_one_patient_out


def run_cohort_leave_one_cohort_out_CV(feature_path=(r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\
                                                     try_0408\Beijing"),
                                       model_base=linear_model.LogisticRegression(class_weight="balanced"),
                                       ML_model_name="LM"):
    grid_point_all = np.load('grid_point_all.npy', allow_pickle='TRUE').item()
    performance_leave_one_cohort_out = {}

    for cohort_test in ["Pittsburgh", "Beijing", "Berlin"]:
        print('cohort: '+str(cohort_test))
        if cohort_test not in performance_leave_one_cohort_out:
            performance_leave_one_cohort_out[cohort_test] = {}

        for grid_point in list(grid_point_all.keys()):
            print('grid point: '+str(grid_point))
            if cohort_test not in grid_point_all[grid_point]:
                continue
            if len(list(grid_point_all[grid_point].keys())) == 1:
                continue  # cannot do leave one cohort prediction with a single cohort

            X_train = []
            y_train = []
            for cohort_train in ["Pittsburgh", "Beijing", "Berlin"]:
                if cohort_test == cohort_train:
                    continue
                if cohort_train not in grid_point_all[grid_point]:
                    continue
                for subject_test in list(grid_point_all[grid_point][cohort_train].keys()):
                    for run in list(grid_point_all[grid_point][cohort_train][subject_test].keys()):
                        if grid_point_all[grid_point][cohort_train][subject_test][run]["lat"] != "CON":
                            continue
                        X_train.append(grid_point_all[grid_point][cohort_train][subject_test][run]["data"])
                        y_train.append(grid_point_all[grid_point][cohort_train][subject_test][run]["label"])
            if len(X_train) > 1:
                X_train = np.concatenate(X_train, axis=0)
                y_train = np.concatenate(y_train, axis=0)
            else:
                X_train = X_train[0]
                y_train = y_train[0]

            # run here ML estimation
            model = clone(model_base)
            if ML_model_name == "XGB":
                X_train, X_val, y_train, y_val = \
                    model_selection.train_test_split(
                        X_train, y_train, train_size=0.7, shuffle=False)
                classes_weights = class_weight.compute_sample_weight(
                    class_weight='balanced', y=y_train)

                model.fit(
                    X_train, y_train, eval_set=[(X_val, y_val)],
                    early_stopping_rounds=7, sample_weight=classes_weights,
                    verbose=False)

            else:
                # LM
                model.fit(X_train, y_train)

            performance_leave_one_cohort_out[cohort_test][grid_point] = {}
            for subject_test in list(grid_point_all[grid_point][cohort_test].keys()):
                X_test = []
                y_test = []
                for run in list(grid_point_all[grid_point][cohort_test][subject_test].keys()):
                    if grid_point_all[grid_point][cohort_test][subject_test][run]["lat"] != "CON":
                        continue
                    X_test.append(grid_point_all[grid_point][cohort_test][subject_test][run]["data"])
                    y_test.append(grid_point_all[grid_point][cohort_test][subject_test][run]["label"])
                if len(X_test) > 1:
                    X_test = np.concatenate(X_test, axis=0)
                    y_test = np.concatenate(y_test, axis=0)
                else:
                    X_test = X_test[0]
                    y_test = y_test[0]

                y_tr_pr = model.predict(X_train)
                y_te_pr = model.predict(X_test)
                performance_leave_one_cohort_out[cohort_test][grid_point][subject_test] = {}
                performance_leave_one_cohort_out[cohort_test][grid_point][subject_test]["y_test"] = y_test
                performance_leave_one_cohort_out[cohort_test][grid_point][subject_test]["y_test_pr"] = y_te_pr
                performance_leave_one_cohort_out[cohort_test][grid_point][subject_test]["y_train"] = y_train
                performance_leave_one_cohort_out[cohort_test][grid_point][subject_test]["y_train_pr"] = y_tr_pr
                performance_leave_one_cohort_out[cohort_test][grid_point][subject_test]["performance_test"] = \
                    metrics.balanced_accuracy_score(y_test, y_te_pr)
                performance_leave_one_cohort_out[cohort_test][grid_point][subject_test]["performance_train"] = \
                    metrics.balanced_accuracy_score(y_train, y_tr_pr)

    # add the cortex grid for plotting
    nm_reader = NM_reader.NM_Reader(feature_path)
    feature_file = nm_reader.get_feature_list()[0]
    grid_cortex = np.array(nm_reader.read_settings(feature_file)["grid_cortex"])
    performance_leave_one_cohort_out["grid_cortex"] = grid_cortex
    np.save(ML_model_name+'_performance_leave_one_cohort_out.npy', performance_leave_one_cohort_out)
