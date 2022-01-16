from re import VERBOSE
import sys
import os
import numpy as np
from pathlib import Path
from scipy import stats
import pandas as pd
from multiprocessing import Pool
from sklearn import linear_model, discriminant_analysis, ensemble, svm
from sklearn import metrics
from sklearn.base import clone
from sklearn import model_selection
from sklearn.utils import class_weight
from scipy.ndimage import (binary_dilation,
                           binary_erosion,
                           label)
import xgboost
import _pickle as cPickle
from scipy import io
from matplotlib import pyplot as plt
import matplotlib
import bids
from bids import BIDSLayout
from itertools import product

import py_neuromodulation
from py_neuromodulation import (
    nm_decode,
    nm_analysis,
    nm_BidsStream,
    nm_IO
)

class CohortRunner:

    def __init__(
        self,
        cohorts:dict = None,
        ML_model_name="LM",
        model=linear_model.LogisticRegression(class_weight="balanced"),
        eval_method=metrics.balanced_accuracy_score,
        estimate_gridpoints=False, 
        estimate_channels=True,
        estimate_all_channels_combined=False,
        save_coef=False,
        TRAIN_VAL_SPLIT=False,
        plot_features=False,
        plot_grid_performances=False,
        run_ML_model=True,
        run_bids=True,
        ECOG_ONLY=True,
        run_pool=True,
        VERBOSE=False,
        LIMIT_DATA=False,
        RUN_BAY_OPT=False,
        cv_method=model_selection.KFold(n_splits=3, shuffle=False),
        use_nested_cv=True,
        outpath=r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\0209_SharpWaveLimFeaturesSTFT_with_Grid",
        PATH_SETTINGS=r"C:\Users\ICN_admin\Documents\py_neuromodulation\pyneuromodulation\nm_settings.json"
    ) -> None:

        self.ML_model_name = ML_model_name
        self.model = model
        self.outpath = outpath
        self.PATH_SETTINGS = PATH_SETTINGS
        self.estimate_gridpoints = estimate_gridpoints
        self.estimate_channels = estimate_channels
        self.estimate_all_channels_combined = estimate_all_channels_combined
        self.save_coef = save_coef
        self.plot_features = plot_features
        self.plot_grid_performances = plot_grid_performances
        self.run_ML_model = run_ML_model
        self.run_bids = run_bids
        self.run_pool = run_pool
        self.ECOG_ONLY = ECOG_ONLY
        self.TRAIN_VAL_SPLIT = TRAIN_VAL_SPLIT
        self.cohorts = cohorts
        self.VERBOSE = VERBOSE
        self.LIMIT_DATA = LIMIT_DATA
        self.eval_method = eval_method
        self.cv_method = cv_method
        self.use_nested_cv = use_nested_cv
        self.RUN_BAY_OPT = RUN_BAY_OPT
        self.TRAIN_VAL_SPLIT = TRAIN_VAL_SPLIT
        self.model = model


        self.grid_cortex = pd.read_csv(
            os.path.join(py_neuromodulation.__path__[0], 'grid_cortex.tsv'), sep='\t'
        ).to_numpy()
    
    def init_decoder(self) -> nm_decode.Decoder:
        return nm_decode.Decoder(
            model=self.model,
            TRAIN_VAL_SPLIT=self.TRAIN_VAL_SPLIT,
            STACK_FEATURES_N_SAMPLES=True,
            get_movement_detection_rate=True,
            eval_method=self.eval_method,
            VERBOSE=self.VERBOSE,
            cv_method=self.cv_method,
            use_nested_cv=self.use_nested_cv,
            RUN_BAY_OPT=self.RUN_BAY_OPT
        )

    def multiprocess_pipeline_run_wrapper(self, PATH_RUN):

        if type(PATH_RUN) is bids.layout.models.BIDSFile:
            PATH_RUN = PATH_RUN.path

        # set BIDS PATH and out path
        # better option: feed the output and bids path as well as a param through the pool

        for cohort, PATH_COHORT in self.cohorts.items():
            if cohort in PATH_RUN:
                PATH_BIDS = PATH_COHORT
                PATH_OUT = os.path.join(self.outpath, cohort)
                break

        if self.run_bids:
            nm_BIDS = nm_BidsStream.BidsStream(
                PATH_RUN=PATH_RUN,
                PATH_BIDS=PATH_BIDS,
                PATH_OUT=PATH_OUT,
                LIMIT_DATA=self.LIMIT_DATA,
                LIMIT_HIGH=200000,
                LIMIT_LOW=0,
                ECOG_ONLY=self.ECOG_ONLY,
                PATH_SETTINGS=self.PATH_SETTINGS,
                VERBOSE=self.VERBOSE
            )

            nm_BIDS.run_bids()

        feature_file = os.path.basename(PATH_RUN)[:-5]  # cut off ".vhdr"

        feature_reader = nm_analysis.Feature_Reader(
                feature_dir=PATH_OUT,
                feature_file=feature_file
        )

        if self.plot_grid_performances:
            feature_reader.plot_cort_projection()

        if self.plot_features:

            ch_to_plot = feature_reader.nm_channels.query(
                '(type=="ecog") and (used == 1)'
            )["name"]

            feature_used = "stft"

            for ch_used in ch_to_plot:
                feature_reader.plot_target_averaged_channel(
                    ch=ch_used,
                    list_feature_keywords=[feature_used],
                    epoch_len=4,
                    threshold=0.5
                )

        #model = discriminant_analysis.LinearDiscriminantAnalysis()
        #model = xgboost.XGBClassifier(scale_pos_weight=10)  # balanced class weights
        #model = ensemble.RandomForestClassifier(n_estimators=6, max_depth=6, class_weight='balanced')
        #model = svm.SVC(class_weight="balanced")

        if self.run_ML_model:
            # set decoder for this specific run (using the feature_reader features)
            feature_reader.decoder = self.init_decoder()

            feature_reader.decoder.set_data(
                features=feature_reader.feature_arr,
                label=feature_reader.label,
                label_name=feature_reader.label_name,
                used_chs=feature_reader.used_chs
            )

            performances = feature_reader.run_ML_model(
                estimate_channels=self.estimate_channels,
                estimate_gridpoints=self.estimate_gridpoints,
                estimate_all_channels_combined=self.estimate_all_channels_combined,
                save_results=True
            )

        if self.plot_grid_performances:
            feature_reader.plot_subject_grid_ch_performance(
                performance_dict=performances,
                plt_grid=True
            )

    def run_cohorts(self):

        run_files_all = []
        for _, PATH_COHORT in self.cohorts.items():
            layout = BIDSLayout(PATH_COHORT)
            run_files_all.append(layout.get(extension='.vhdr'))

        run_files_all = list(np.concatenate(run_files_all))

        if self.run_pool:
            pool = Pool(processes=50)
            pool.map(self.multiprocess_pipeline_run_wrapper, run_files_all)
        else:
            #self.multiprocess_pipeline_run_wrapper(run_files_all[11])
            for run_file in run_files_all[:12]:
                self.multiprocess_pipeline_run_wrapper(run_file)

    def read_cohort_results(self, feature_path, cohort):
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

        # Here the runs are overwritten!
        folders_path = [x[0] for x in os.walk(feature_path)]
        feature_paths = [os.path.basename(x) for x in folders_path[1:]]
        performance_out = {}

        for feature_file in feature_paths:
            feature_reader = nm_analysis.Feature_Reader(
                feature_dir=feature_path,
                feature_file=feature_file
            )

            performance_run = feature_reader.read_results(
                read_grid_points=self.estimate_gridpoints,
                read_channels=self.estimate_channels,
                read_all_combined=self.estimate_all_channels_combined,
                read_mov_detection_rates=True
            )

            sub = feature_file[feature_file.find('sub-'):feature_file.find('_ses')][4:]
            if sub not in performance_out:
                performance_out[sub] = {}
            performance_out[sub][feature_file] = performance_run[sub]  # get saved in performance_run

        np.save(os.path.join(self.outpath, self.ML_model_name+'_cohort_'+cohort+'.npy'),
            performance_out
        )

    def cohort_wrapper_read_cohort(self):
        """Read results for multiple cohorts"""

        for cohort in self.cohorts.keys():
            self.read_cohort_results(os.path.join(self.outpath, cohort), cohort)

    def read_all_channels(
        self,
        channel_all,
        feature_path,
        feature_file,
        cohort,
        read_channels : bool = True
    ):
        """Save for a given feature path all used grid point data. Necessary to run across patient and cohort analysis.
        Parameters
        ----------
        channel_all : dictionary
            dictionary with data, label, label_name and feature_names for each channel
        feature_path : string
            path to feature files
        feature_file : string
            feature file
        cohort : string
            used for indecing of grid_point_all
        read_ch : bool
            if True read channels, else read grid_points
        Returns
        -------
        dictionary
            ch_all
        """
        feature_wrapper = nm_analysis.FeatureReadWrapper(
            feature_dir=feature_path,
            feature_file=feature_file
        )
        decoder = nm_decode.Decoder(feature_path=feature_path,
                                    feature_file=feature_file)
        decoder.label = feature_wrapper.nm_reader.label
        decoder.target_ch = feature_wrapper.label_name
        subject_name = feature_file[feature_file.find("sub-")+4:feature_file.find("_ses")]
        sess_name = feature_file[feature_file.find("ses-")+4:feature_file.find("_task")]
        task_name = feature_file[feature_file.find("task-")+5:feature_file.find("_run")]
        run_number = feature_file[feature_file.find("run-")+4:feature_file.find("_ieeg")]

        if read_channels is True:
            decoder.set_data_ind_channels()
            data_to_read = decoder.ch_ind_data
        else:
            decoder.set_data_grid_points()
            data_to_read = decoder.grid_point_ind_data

        for ch in list(data_to_read.keys()):
            if cohort not in channel_all:
                channel_all[cohort] = {}
            if subject_name not in channel_all[cohort]:
                channel_all[cohort][subject_name] = {}
            if ch not in channel_all[cohort][subject_name]:
                channel_all[cohort][subject_name][ch] = {}
            channel_all[cohort][subject_name][ch][feature_file] = {}

            channel_all[cohort][subject_name][ch][feature_file]["data"] = data_to_read[ch]
            channel_all[cohort][subject_name][ch][feature_file]["feature_names"] = \
                [ch_[len(ch)+1:] for ch_ in decoder.features.columns if ch in ch_]
            channel_all[cohort][subject_name][ch][feature_file]["label"] = decoder.label
            channel_all[cohort][subject_name][ch][feature_file]["label_name"] = decoder.target_ch

            # check laterality
            lat = "CON"  # Beijing is always contralateral
            # Pittsburgh Subjects
            if ("LEFT" in decoder.target_ch and "LEFT" in decoder.features.columns[1]) or \
            ("RIGHT" in decoder.target_ch and "RIGHT" in decoder.features.columns[1]):
                lat = "IPS"

            # Berlin subjects
            if ("_L_" in decoder.features.columns[1] and task_name == "SelfpacedRotationL") or \
            ("_R_" in decoder.features.columns[1] and task_name == "SelfpacedRotationR"):
                lat = "IPS"
            channel_all[cohort][subject_name][ch][feature_file]["lat"] = lat
        return channel_all

    def cohort_wrapper_read_all_grid_points(self, read_channels=True, feature_path_cohorts=\
            r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\0209_SharpWaveLimFeaturesSTFT_with_Grid"):
        cohorts = self.cohorts.keys()
        grid_point_all = {}
        for cohort in cohorts:
            print("COHORT: "+cohort)
            feature_path = os.path.join(feature_path_cohorts, cohort)
            feature_list = nm_IO.get_run_list_indir(feature_path)
            for feature_file in feature_list:
                print(feature_file)
                grid_point_all = self.read_all_grid_points(
                    grid_point_all,
                    feature_path,
                    feature_file,
                    cohort,
                    read_channels=read_channels
                )

        if read_channels is True:
            np.save(os.path.join(self.outpath, 'channel_all.npy'), grid_point_all)
        else:
            np.save(os.path.join(self.outpath, 'grid_point_all.npy'), grid_point_all)

    def eval_model(self, X_train, y_train, X_test, y_test):

        return self.decoder.wrapper_model_train(
            X_train,
            y_train,
            X_test,
            y_test,
            cv_res=nm_decode.CV_res()
        )


    def run_cohort_leave_one_patient_out_CV_within_cohort(self,
            feature_path=r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\try_0408\Beijing"):

        grid_point_all = np.load(os.path.join(feature_path, 'grid_point_all.npy'), allow_pickle='TRUE').item()
        performance_leave_one_patient_out = {}

        for cohort in self.cohorts:
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

                    # run here ML estimation
                    self.decoder = self.init_decoder()
                    model = self.decoder.wrapper_model_train(
                        X_train=X_train,
                        y_train=y_train,
                        return_fitted_model_only=True
                    )
                    # use initialized decoder
                    try:
                        cv_res = self.eval_model(X_train, y_train, X_test, y_test)
                    except nm_decode.Decoder.ClassMissingException:
                        continue
    
                    performance_leave_one_patient_out[cohort][grid_point][subject_test] = cv_res

        performance_leave_one_patient_out["grid_cortex"] = self.grid_cortex
        np.save(os.path.join(self.outpath, self.ML_model_name+'_performance_leave_one_patient_out_within_cohort.npy'),\
            performance_leave_one_patient_out)
        return performance_leave_one_patient_out

    def run_cohort_leave_one_cohort_out_CV(self,
        feature_path=r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\try_0408\Beijing"):
        grid_point_all = np.load(os.path.join(feature_path, 'grid_point_all.npy'), allow_pickle='TRUE').item()
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
                self.decoder = self.init_decoder()
                model = self.decoder.wrapper_model_train(
                    X_train=X_train,
                    y_train=y_train,
                    return_fitted_model_only=True
                )

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

                    cv_res = self.decoder.eval_model(
                        model,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        cv_res = nm_decode.CV_res()
                    )

                    performance_leave_one_cohort_out[cohort_test][grid_point][subject_test] = cv_res

        performance_leave_one_cohort_out["grid_cortex"] = self.grid_cortex
        np.save(os.path.join(self.outpath, self.ML_model_name+'_performance_leave_one_cohort_out.npy'), performance_leave_one_cohort_out)

    def run_leave_one_patient_out_across_cohorts(
        self,
        feature_path=r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\try_0408\Beijing"
    ):

        grid_point_all = np.load(os.path.join(feature_path, 'grid_point_all.npy'), allow_pickle='TRUE').item()
        performance_leave_one_patient_out = {}

        for grid_point in list(grid_point_all.keys()):
            print('grid point: '+str(grid_point))
            for cohort in ["Pittsburgh", "Beijing", "Berlin"]:
                print('cohort: '+str(cohort))
                if cohort not in performance_leave_one_patient_out:
                    performance_leave_one_patient_out[cohort] = {}

                if cohort not in grid_point_all[grid_point]:
                    continue
                if len(list(grid_point_all[grid_point][cohort].keys())) <= 1:
                    continue  # cannot do leave one out prediction with a single subject
                
                if grid_point not in performance_leave_one_patient_out[cohort]:
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
                    for cohort_inner in list(grid_point_all[grid_point].keys()):  # available cohorts for that grid point
                        for subject_train in list(grid_point_all[grid_point][cohort_inner].keys()):
                            if (subject_test == subject_train) and (cohort_inner == cohort):
                                continue
                            for run in list(grid_point_all[grid_point][cohort_inner][subject_train].keys()):
                                if grid_point_all[grid_point][cohort_inner][subject_train][run]["lat"] != "CON":
                                    continue
                                X_train.append(grid_point_all[grid_point][cohort_inner][subject_train][run]["data"])
                                y_train.append(grid_point_all[grid_point][cohort_inner][subject_train][run]["label"])
                    if len(X_train) > 1:
                        X_train = np.concatenate(X_train, axis=0)
                        y_train = np.concatenate(y_train, axis=0)
                    else:
                        X_train = X_train[0]
                        y_train = y_train[0]

                    self.decoder = self.init_decoder()
                    try:
                        cv_res = self.eval_model(X_train, y_train, X_test, y_test)
                    except nm_decode.Decoder.ClassMissingException:
                        continue
    
                    performance_leave_one_patient_out[cohort][grid_point][subject_test] = cv_res

        performance_leave_one_patient_out["grid_cortex"] = self.grid_cortex
        np.save(
            os.path.join(
                self.outpath,
                self.ML_model_name+'_performance_leave_one_patient_out_across_cohorts.npy'
            ),
            performance_leave_one_patient_out
        )

    def run_leave_nminus1_patient_out_across_cohorts(
        self,
        feature_path=r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\try_0408\Beijing"
    ):

        grid_point_all = np.load(os.path.join(feature_path, 'grid_point_all.npy'), allow_pickle='TRUE').item()
        performance_leave_one_patient_out = {}

        for grid_point in list(grid_point_all.keys()):
            print('grid point: '+str(grid_point))
            for cohort_train in ["Pittsburgh", "Beijing", "Berlin"]:
                print('cohort: '+str(cohort_train))
                if cohort_train not in performance_leave_one_patient_out:
                    performance_leave_one_patient_out[cohort_train] = {}

                if cohort_train not in grid_point_all[grid_point]:
                    continue
                if len(list(grid_point_all[grid_point][cohort_train].keys())) <= 1:
                    continue  # cannot do leave one out prediction with a single subject
                if grid_point not in performance_leave_one_patient_out[cohort_train]:
                    performance_leave_one_patient_out[cohort_train][grid_point] = {}

                for subject_train in list(grid_point_all[grid_point][cohort_train].keys()):
                    X_train = []
                    y_train = []
                    for run in list(grid_point_all[grid_point][cohort_train][subject_train].keys()):
                        if grid_point_all[grid_point][cohort_train][subject_train][run]["lat"] != "CON":
                            continue
                        X_train.append(grid_point_all[grid_point][cohort_train][subject_train][run]["data"])
                        y_train.append(grid_point_all[grid_point][cohort_train][subject_train][run]["label"])
                    if len(X_train) > 1:
                        X_train = np.concatenate(X_train, axis=0)
                        y_train = np.concatenate(y_train, axis=0)
                    else:
                        X_train = X_train[0]
                        y_train = y_train[0]

                    for cohort_test in list(grid_point_all[grid_point].keys()):
                        for subject_test in list(grid_point_all[grid_point][cohort_test].keys()):
                            if (subject_test == subject_train) and (cohort_test == cohort_train):
                                continue
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

                            self.decoder = self.init_decoder()
                            try:
                                cv_res = self.eval_model(X_train, y_train, X_test, y_test)
                            except nm_decode.Decoder.ClassMissingException:
                                continue

                            if subject_train not in performance_leave_one_patient_out[cohort_train][grid_point]:
                                performance_leave_one_patient_out[cohort_train][grid_point][subject_train] = {}
                            if cohort_test not in performance_leave_one_patient_out[cohort_train][grid_point][subject_train]:
                                performance_leave_one_patient_out[cohort_train][grid_point][subject_train][cohort_test] = {}
                            if subject_test not in performance_leave_one_patient_out[cohort_train][grid_point][subject_train][cohort_test]:
                                performance_leave_one_patient_out[cohort_train][grid_point][subject_train][cohort_test][subject_test] = {}

                            performance_leave_one_patient_out[cohort_train][grid_point][subject_train][cohort_test][subject_test] = cv_res

        performance_leave_one_patient_out["grid_cortex"] = self.grid_cortex
        np.save(os.path.join(self.outpath, self.ML_model_name+'_performance_leave_nminus1_patient_out_across_cohorts.npy'), performance_leave_one_patient_out)
