import sys
import os
import numpy as np
from pathlib import Path
from scipy import stats
import multiprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
import xgboost
import _pickle as cPickle
from scipy import io
from matplotlib import pyplot as plt
import matplotlib
import bids
from bids import BIDSLayout

from pyneuromodulation import nm_reader as NM_reader
from pyneuromodulation import nm_decode
from pyneuromodulation import start_BIDS
from pyneuromodulation import settings as nm_settings


class FeatureReadWrapper:

    def __init__(self, feature_path="C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\write_out\\try_0408",
                 feature_file=None, plt_cort_projection=False,
                 read_features=True) -> None:
        """FeatureReadWrapper enables analysis methods on top of NM_reader and NM_Decoder

        Parameters
        ----------
        feature_path : str, optional
            Path to py_neuromodulation estimated feature runs, where each feature is a folder,
            by default "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\write_out\\try_0408"
        feature_file : str, optional
            specific feature run, if None it is set to the first feature folder in feature_path
        plt_cort_projection : bool, optional
            if true, calls nm_reader plot_cortical_projection, by default False
        read_features : bool, optional
            if true, read features.csv filde and extract label, the label is assumed to be binary,
            by default True
        """
        self.feature_path = feature_path
        self.nm_reader = NM_reader.NM_Reader(feature_path)
        self.feature_list = self.nm_reader.get_feature_list()
        if feature_file is None:
            self.feature_file = self.feature_list[0]
        else:
            self.feature_file = feature_file

        self.settings = self.nm_reader.read_settings(feature_file)
        self.ch_names = self.settings["ch_names"]
        self.ch_names_ECOG = [ch_name for ch_name in self.ch_names if "ECOG" in ch_name]

        # read run_analysis
        self.run_analyzer = self.nm_reader.read_run_analyzer()

        self.PATH_PLOT = os.path.abspath("plots")
        self.nm_reader.read_plot_modules(self.PATH_PLOT)
        if plt_cort_projection is True:
            self.nm_reader.plot_cortical_projection()

        _ = self.nm_reader.read_M1(self.feature_file)

        target_names = list(self.nm_reader.df_M1[self.nm_reader.df_M1["target"] == 1]["name"])
        target_clean = [target for target in target_names if ("clean" in target) or
                        ("squared" in target) or ("CLEAN" in target)]
        if len(target_clean) == 0:
            target = target_names[0]
        else:
            for target_ in target_clean:
                # try to select contralateral label
                if self.settings["sess_right"] is True and "LEFT" in target_:
                    target = target_
                    continue
                elif self.settings["sess_right"] is False and "RIGHT" in target_:
                    target = target_
                    continue
                if target_ == target_clean[-1]:
                    target = target_clean[0]  # set label to last element
        self.label_name = target

        if read_features is True:
            _ = self.nm_reader.read_features(self.feature_file)
            self.nm_reader.label = np.nan_to_num(np.array(self.nm_reader.read_label(target))) > 0.3

    def read_plotting_modules(self):
        self.faces = io.loadmat(os.path.join(self.PATH_PLOT, 'faces.mat'))
        self.vertices = io.loadmat(os.path.join(self.PATH_PLOT, 'Vertices.mat'))
        self.grid = io.loadmat(os.path.join(self.PATH_PLOT, 'grid.mat'))['grid']
        self.stn_surf = io.loadmat(os.path.join(self.PATH_PLOT, 'STN_surf.mat'))
        self.x_ver = self.stn_surf['vertices'][::2, 0]
        self.y_ver = self.stn_surf['vertices'][::2, 1]
        self.x_ecog = self.vertices['Vertices'][::1, 0]
        self.y_ecog = self.vertices['Vertices'][::1, 1]
        self.z_ecog = self.vertices['Vertices'][::1, 2]
        self.x_stn = self.stn_surf['vertices'][::1, 0]
        self.y_stn = self.stn_surf['vertices'][::1, 1]
        self.z_stn = self.stn_surf['vertices'][::1, 2]

    def plot_subject_grid_ch_performance(self, sub, performance_dict=None, plt_grid=False):
        """plot subject specific performance for individual channeal and optional grid points

        Parameters
        ----------
        sub : string
            used subject
        performance_dict : dict, optional
            [description], by default None
        plt_grid : bool, optional
            True to plot grid performances, by default False
        """
        # if performance_dict is None:
        #    with open(r'C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\META\Beijing_out.p', 'rb') as input:
        #        performance_dict = cPickle.load(input)

        ecog_strip_performance = []
        ecog_coords_strip = []
        cortex_grid = []
        grid_performance = []

        ch_ = list(performance_dict[sub].keys())
        for ch in ch_:
            if 'grid_' not in ch:
                ecog_coords_strip.append(performance_dict[sub][ch]["coord"])
                ecog_strip_performance.append(performance_dict[sub][ch]["performance"])
            elif plt_grid is True and 'grid_' in ch:
                cortex_grid.append(performance_dict[sub][ch]["coord"])
                grid_performance.append(performance_dict[sub][ch]["performance"])

        ecog_coords_strip = np.vstack(ecog_coords_strip).T
        if ecog_coords_strip[0, 0] > 0:  # it's on the right side GIVEN IT'S CONTRALATERAL
            ecog_coords_strip[0, :] *= -1

        fig, axes = plt.subplots(1, 1, facecolor=(1, 1, 1),
                                 figsize=(14, 9))  # , dpi=300)
        axes.scatter(self.x_ecog, self.y_ecog, c="gray", s=0.001)
        axes.axes.set_aspect('equal', anchor='C')

        pos_elec = axes.scatter(ecog_coords_strip[0, :],
                                ecog_coords_strip[1, :], c=ecog_strip_performance,
                                s=50, alpha=0.8, cmap="viridis", marker="x")

        if plt_grid is True:
            cortex_grid = np.array(cortex_grid).T
            pos_ecog = axes.scatter(cortex_grid[0, :],
                                    cortex_grid[1, :], c=grid_performance,
                                    s=30, alpha=0.8, cmap="viridis")

        plt.axis('off')
        pos_elec.set_clim(0.5, 0.8)
        cbar = fig.colorbar(pos_elec)
        cbar.set_label("Balanced Accuracy")

        PATH_SAVE = os.path.join(self.feature_path, self.feature_file, 'grid_channel_performance.png')
        plt.savefig(PATH_SAVE, bbox_inches="tight")
        print("saved Figure to : " + str(PATH_SAVE))

    def plot_features_per_channel(self, ch_name,
                                  plt_corr_matr=False, feature_file=None):

        if feature_file is not None:
            self.feature_file = feature_file
        # Fist case: filter for bandpass activity features only
        dat_ch = self.nm_reader.read_channel_data(ch_name, read_bp_activity_only=True)

        # estimating epochs, with shape (epochs,samples,channels,features)
        X_epoch, y_epoch = self.nm_reader.get_epochs_ch(epoch_len=4,
                                                        sfreq=self.settings["sampling_rate_features"],
                                                        threshold=0.1)
        if plt_corr_matr is True:
            print("plotting feature covariance matrix")
            self.nm_reader.plot_corr_matrix(self.feature_file, feature_str_add="bandpass")
        print("plotting feature target averaged")
        self.nm_reader.plot_epochs_avg(self.feature_file, feature_str_add="bandpass")

        # Second case: filter for sharpwave prominence features only
        dat_ch = self.nm_reader.read_channel_data(ch_name, read_sharpwave_prominence_only=True)

        # estimating epochs, with shape (epochs,samples,channels,features)
        X_epoch, y_epoch = self.nm_reader.get_epochs_ch(epoch_len=4,
                                                        sfreq=self.settings["sampling_rate_features"],
                                                        threshold=0.1)
        if plt_corr_matr is True:
            print("plotting feature covariance matrix")
            self.nm_reader.plot_corr_matrix(self.feature_file,
                                            feature_str_add="sharpwaveprominence")
        print("plotting feature target averaged")
        self.nm_reader.plot_epochs_avg(self.feature_file,
                                       feature_str_add="sharpwaveprominence")

    def plot_features(self):
        for ch_name_ECOG in self.ch_names_ECOG:
            self.plot_features_per_channel(ch_name_ECOG, plt_corr_matr=False)

    def run_ML_LM(self, feature_file=None):

        if feature_file is not None:
            self.feature_file = feature_file

        model = linear_model.LogisticRegression(class_weight="balanced")
        decoder = nm_decode.Decoder(feature_path=self.feature_path,
                                    feature_file=self.feature_file,
                                    model=model,
                                    eval_method=metrics.balanced_accuracy_score,
                                    cv_method=model_selection.KFold(n_splits=3,
                                                                    shuffle=False),
                                    threshold_score=True
                                    )
        decoder.label = self.nm_reader.label
        decoder.target_ch = self.label_name  # label name
        # run estimations for channels and grid points individually
        # currently the MLp file get's saved only, and overwrited previous files
        decoder.set_data_ind_channels()
        decoder.set_data_grid_points()

        decoder.run_CV_ind_channels(XGB=False)
        decoder.run_CV_grid_points(XGB=False)
        decoder.save("LM")

    def read_ind_channel_results(self, performance_dict, subject_name=None,
                                 feature_file=None, DEFAULT_PERFORMANCE=0.5):
        if feature_file is not None:
            self.feature_file = feature_file

        PATH_ML_ = os.path.join(self.feature_path, self.feature_file,
                                self.feature_file + "_LM_ML_RES.p")

        # read ML results
        with open(PATH_ML_, 'rb') as input:
            ML_res = cPickle.load(input)

        performance_dict[subject_name] = {}

        # channels
        ch_to_use = list(np.array(ML_res.settings["ch_names"])
                         [np.where(np.array(ML_res.settings["ch_types"]) == 'ecog')[0]])
        for ch in ch_to_use:

            performance_dict[subject_name][ch] = {}  # should be 7 for Berlin

            if ML_res.settings["sess_right"] is True:
                cortex_name = "cortex_right"
            else:
                cortex_name = "cortex_left"

            idx_ = [idx for idx, i in enumerate(ML_res.settings["coord"][cortex_name]["ch_names"])
                    if ch.startswith(i+'-')][0]
            coords = ML_res.settings["coord"][cortex_name]["positions"][idx_]
            performance_dict[subject_name][ch]["coord"] = coords
            performance_dict[subject_name][ch]["performance"] = np.mean(ML_res.ch_ind_pr[ch]["score_test"])

        # read now also grid point results
        performance_dict[subject_name]["active_gridpoints"] = ML_res.active_gridpoints
        for grid_point in range(len(ML_res.settings["grid_cortex"])):
            performance_dict[subject_name]["grid_"+str(grid_point)] = {}
            performance_dict[subject_name]["grid_"+str(grid_point)]["coord"] = \
                ML_res.settings["grid_cortex"][grid_point]
            if grid_point in ML_res.active_gridpoints:
                performance_dict[subject_name]["grid_"+str(grid_point)]["performance"] = \
                    np.mean(ML_res.gridpoint_ind_pr[grid_point]["score_test"])
            else:
                # set non interpolated grid point to default performance
                performance_dict[subject_name]["grid_"+str(grid_point)]["performance"] = DEFAULT_PERFORMANCE
        return performance_dict


def multiprocess_pipeline_run_wrapper(PATH_RUN):

    if type(PATH_RUN) is bids.layout.models.BIDSFile:
        PATH_RUN = PATH_RUN.path

    settings_wrapper = nm_settings.SettingsWrapper(settings_path='settings.json')

    start_BIDS.est_features_run(PATH_RUN)
    feature_path = settings_wrapper.settings["out_path"]
    feature_file = os.path.basename(PATH_RUN)[:-5]  # cut off ".vhdr"

    feature_wrapper = FeatureReadWrapper(feature_path, feature_file,
                                         plt_cort_projection=True)
    feature_wrapper.read_plotting_modules()

    feature_wrapper.plot_features()
    feature_wrapper.run_ML_LM()

    performance_dict = {}

    # subject_name is different across cohorts
    subject_name = feature_wrapper.feature_file[4:10]
    performance_dict = feature_wrapper.read_ind_channel_results(performance_dict,
                                                                subject_name)

    feature_wrapper.plot_subject_grid_ch_performance(subject_name,
                                                     performance_dict=performance_dict,
                                                     plt_grid=True)


def run_cohort(cohort="Pittsburgh"):

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

    # check OPTIONALLY for Pittsburgh which ones where not run through yet, and run those
    run_files_left = []
    directory = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\try_0408\Pittsburgh"
    folders_path = [x[0] for x in os.walk(directory)]
    folders = [os.path.basename(x) for x in folders_path[1:]]
    for run_file in run_files:
        feature_file = os.path.basename(run_file.path)[:-5]
        if feature_file not in folders:
            run_files_left.append(run_file)

    # multiprocess_pipeline_run_wrapper(run_files[0])
    pool = multiprocessing.Pool(processes=55)  # most on Ryzen 2990WX is 63
    pool.map(multiprocess_pipeline_run_wrapper, run_files_left)


def read_cohort(feature_path, cohort):

    folders_path = [x[0] for x in os.walk(feature_path)]
    feature_paths = [os.path.basename(x) for x in folders_path[1:]]
    performance_dict = {}

    for feature_file in feature_paths:
        feature_wrapper = FeatureReadWrapper(feature_path, feature_file,
                                             plt_cort_projection=False, read_features=False)
        subject_name = feature_file[feature_file.find("sub-"):feature_file.find("_ses")]
        # cut here s.t. the subject is the whole recording
        subject_name = feature_file[:-5]
        performance_dict = feature_wrapper.read_ind_channel_results(performance_dict,
                                                                    subject_name)
    np.save('cohort_'+cohort+'.npy', performance_dict)


def cohort_wrapper_read_cohort():
    cohorts = ["Pittsburgh", "Beijing", "Berlin"]
    feature_path = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\try_0408"
    for cohort in cohorts:
        read_cohort(os.path.join(feature_path, cohort), cohort)


def read_all_grid_points(grid_point_all, feature_path, feature_file, cohort):

    feature_wrapper = FeatureReadWrapper(feature_path, feature_file,
                                         plt_cort_projection=False)

    model = linear_model.LogisticRegression(class_weight="balanced")
    decoder = nm_decode.Decoder(feature_path=feature_path,
                                feature_file=feature_file,
                                model=model,
                                eval_method=metrics.balanced_accuracy_score,
                                cv_method=model_selection.KFold(n_splits=3,
                                                                shuffle=False),
                                threshold_score=True
                                )
    decoder.label = feature_wrapper.nm_reader.label
    decoder.target_ch = feature_wrapper.label_name  # label name
    # run estimations for channels and grid points individually
    # currently the MLp file get's saved only, and overwrited previous files
    # decoder.set_data_ind_channels()
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


def cohort_wrapper_read_all_grid_points():
    cohorts = ["Pittsburgh", "Beijing", "Berlin"]
    feature_path_cohorts = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\try_0408"
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


def run_cohort_leave_one_patient_out_CV():

    grid_point_all = np.load('grid_point_all.npy', allow_pickle='TRUE').item()
    performance_leave_one_patient_out = {}

    for cohort in ["Pittsburgh", "Beijing", "Berlin"]:
        performance_leave_one_patient_out[cohort] = {}

        for grid_point in list(grid_point_all.keys()):
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
                if len(X_test) > 1:
                    X_train = np.concatenate(X_train, axis=0)
                    y_train = np.concatenate(y_train, axis=0)
                else:
                    X_train = X_train[0]
                    y_train = y_train[0]

                # run here ML estimation
                model = linear_model.LogisticRegression(class_weight="balanced")
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
    feature_path = "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\write_out\\try_0408\\Beijing"
    nm_reader = NM_reader.NM_Reader(feature_path)
    feature_file = nm_reader.get_feature_list()[0]
    grid_cortex = np.array(nm_reader.read_settings(feature_file)["grid_cortex"])
    performance_leave_one_patient_out["grid_cortex"] = grid_cortex
    np.save('performance_leave_one_patient_out.npy', performance_leave_one_patient_out)
    return performance_leave_one_patient_out


def run_cohort_leave_one_cohort_out_CV():
    grid_point_all = np.load('grid_point_all.npy', allow_pickle='TRUE').item()
    performance_leave_one_cohort_out = {}

    for cohort_test in ["Pittsburgh", "Beijing", "Berlin"]:

        if cohort_test not in performance_leave_one_cohort_out:
            performance_leave_one_cohort_out[cohort_test] = {}

        for grid_point in list(grid_point_all.keys()):
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
            model = linear_model.LogisticRegression(class_weight="balanced")
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
    feature_path = "C:\\Users\\ICN_admin\\Documents\\Decoding_Toolbox\\write_out\\try_0408\\Beijing"
    nm_reader = NM_reader.NM_Reader(feature_path)
    feature_file = nm_reader.get_feature_list()[0]
    grid_cortex = np.array(nm_reader.read_settings(feature_file)["grid_cortex"])
    performance_leave_one_cohort_out["grid_cortex"] = grid_cortex
    np.save('performance_leave_one_cohort_out.npy', performance_leave_one_cohort_out)


if __name__ == "__main__":

    #run_cohort_leave_one_patient_out_CV()
    run_cohort_leave_one_cohort_out_CV()
    # run single run
    # PATH_RUN = r"C:\Users\ICN_admin\Documents\Decoding_Toolbox\Data\Pittsburgh\sub-000\ses-right\ieeg\sub-000_ses-right_task-force_run-3_ieeg.vhdr"
    # multiprocess_pipeline_run_wrapper(PATH_RUN)

