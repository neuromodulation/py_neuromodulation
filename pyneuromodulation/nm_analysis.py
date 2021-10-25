import os
from scipy import io
from matplotlib import pyplot as plt
import numpy as np
import bids
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
import _pickle as cPickle

from pyneuromodulation import nm_reader as NM_reader
from pyneuromodulation import nm_decode


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

        self.settings = self.nm_reader.read_settings(self.feature_file)
        self.ch_names = self.settings["ch_names"]
        self.ch_names_ECOG = [ch_name for ch_name in self.ch_names if "ECOG" in ch_name]

        # read run_analysis
        self.run_analyzer = self.nm_reader.read_run_analyzer()

        self.PATH_PLOT = os.path.abspath("plots")
        self.nm_reader.read_plot_modules(self.PATH_PLOT)
        self.read_plotting_modules()
        if plt_cort_projection is True:
            self.nm_reader.plot_cortical_projection()

        _ = self.nm_reader.read_nm_channels(self.feature_file)

        target_names = list(self.nm_reader.nm_channels[self.nm_reader.nm_channels["target"] == 1]["name"])
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
        """read necessary mat files for cortex and STN
        """

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

    def plot_subject_grid_ch_performance(self, subject_name=None, performance_dict=None,
                                         plt_grid=False, output_name="LM"):
        """plot subject specific performance for individual channeal and optional grid points

        Parameters
        ----------
        sub : string, optional
            used subject, by default None
        performance_dict : dict, optional
            [description], by default None
        plt_grid : bool, optional
            True to plot grid performances, by default False
        output_name : string, optional
            figure output_name, by default "LM"
        """
        # if performance_dict is None:
        #    with open(r'C:\Users\ICN_admin\Documents\Decoding_Toolbox\write_out\META\Beijing_out.p', 'rb') as input:
        #        performance_dict = cPickle.load(input)

        ecog_strip_performance = []
        ecog_coords_strip = []
        cortex_grid = []
        grid_performance = []

        if subject_name is None:
            subject_name = self.feature_file[self.feature_file.find('sub-'):self.feature_file.find('_ses')][4:]

        ch_ = list(performance_dict[subject_name].keys())
        for ch in ch_:
            if 'grid' not in ch:
                ecog_coords_strip.append(performance_dict[subject_name][ch]["coord"])
                ecog_strip_performance.append(performance_dict[subject_name][ch]["performance_test"])
            elif plt_grid is True and 'grid_' in ch:
                cortex_grid.append(performance_dict[subject_name][ch]["coord"])
                grid_performance.append(performance_dict[subject_name][ch]["performance_test"])

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

        PATH_SAVE = os.path.join(self.feature_path, self.feature_file,
                                 output_name+'_grid_channel_performance.png')
        plt.savefig(PATH_SAVE, bbox_inches="tight")
        print("saved Figure to : " + str(PATH_SAVE))

    def plot_features_per_channel(self, ch_name,
                                  plt_corr_matr=False, plt_stft_features=True,
                                  plt_fft_features=False, plt_bandfiltvar=False,
                                  plt_sharpwave=False, feature_file=None):
        """

        Parameters
        ----------
        ch_name : string
            channel name, as referred in features.csv
        plt_corr_matr : bool, optional
            if True plot correlation matrix for sharpwave and bandpower features, by default False
        plt_stft_features : bool, optional
            if True plot stft movement averaged features, by default True
        plt_fft_features : bool, optional
            if True plot fft movement averaged features, by default True
        plt_bandfiltvar : bool, optional
            if True plot bandpass filtered movement averaged features, by default False
        plt_sharpwave : bool, optional
            if True plot sharpwave movement averaged features, by default False
        feature_file : string, optional
            py_neuromodulation estimated feature file, by default None
        """

        if feature_file is not None:
            self.feature_file = feature_file

        if plt_bandfiltvar:
            dat_ch = self.nm_reader.read_channel_data(ch_name, ['bandpass', 'activity'])

            # estimating epochs, with shape (epochs,samples,channels,features)
            X_epoch, y_epoch = self.nm_reader.get_epochs_ch(epoch_len=4,
                                                            sfreq=self.settings["sampling_rate_features"],
                                                            threshold=0.1)
            print("plotting feature target averaged")
            self.nm_reader.plot_epochs_avg(self.feature_file, feature_str_add="bandpass")

            if plt_corr_matr is True:
                print("plotting feature covariance matrix")
                self.nm_reader.plot_corr_matrix(self.feature_file, feature_str_add="bandpass")

        if plt_stft_features:
            # plot STFT
            dat_ch = self.nm_reader.read_channel_data(ch_name, ['stft'])  # regex needs to be list
            # estimating epochs, with shape (epochs,samples,channels,features)
            X_epoch, y_epoch = self.nm_reader.get_epochs_ch(epoch_len=4,
                                                            sfreq=self.settings["sampling_rate_features"],
                                                            threshold=0.1)
            self.nm_reader.plot_epochs_avg(self.feature_file, feature_str_add='stft')

        if plt_sharpwave:
            dat_ch = self.nm_reader.read_channel_data(ch_name, ['Sharpwave'])

            # estimating epochs, with shape (epochs,samples,channels,features)
            X_epoch, y_epoch = self.nm_reader.get_epochs_ch(epoch_len=4,
                                                            sfreq=self.settings["sampling_rate_features"],
                                                            threshold=0.1)
            if plt_corr_matr is True:
                print("plotting feature covariance matrix")
                self.nm_reader.plot_corr_matrix(self.feature_file,
                                                feature_str_add="sharpwave")
            print("plotting feature target averaged")
            self.nm_reader.plot_epochs_avg(self.feature_file,
                                           feature_str_add="sharpwave")
        if plt_fft_features:
            dat_ch = self.nm_reader.read_channel_data(ch_name, ['fft'])  # regex needs to be list
            # estimating epochs, with shape (epochs,samples,channels,features)
            X_epoch, y_epoch = self.nm_reader.get_epochs_ch(epoch_len=4,
                                                            sfreq=self.settings["sampling_rate_features"],
                                                            threshold=0.1)
            self.nm_reader.plot_epochs_avg(self.feature_file, feature_str_add='fft')

    def plot_features(self, ch_names_ECOG=None):
        """Wrapper that call plot_features_per_channel for every given ECoG channel

        Parameters
        ----------
        ch_names_ECOG : list, optional
            list of ECoG channel to plot features for, by default None
        """
        if ch_names_ECOG is None:
            ch_names_ECOG = self.ch_names_ECOG
        for ch_name_ECOG in ch_names_ECOG:
            self.plot_features_per_channel(ch_name_ECOG, plt_corr_matr=False,
                                           plt_stft_features=False,
                                           plt_sharpwave=False,
                                           plt_fft_features=True)

    def run_ML_model(self, feature_file=None, estimate_gridpoints=True, estimate_channels=True,
                     estimate_all_channels_combined=False,
                     model=linear_model.LogisticRegression(class_weight="balanced"),
                     eval_method=metrics.balanced_accuracy_score,
                     cv_method=model_selection.KFold(n_splits=3, shuffle=False),
                     output_name="LM", TRAIN_VAL_SPLIT=False,
                     save_coef=False):
        """machine learning model evaluation for ECoG strip channels and/or grid points

        Parameters
        ----------
        feature_file : string, optional
            [description], by default None
        estimate_gridpoints : bool, optional
            run ML analysis for grid points, by default True
        estimate_channels : bool, optional
            run ML analysis for ECoG strip channel, by default True
        estimate_all_channels_combined : bool, optional
            run ML analysis features of all channels concatenated, by default False
        model : sklearn model, optional
            ML model, needs to obtain fit and predict functions,
            by default linear_model.LogisticRegression(class_weight="balanced")
        eval_method : sklearn.metrics, optional
            evaluation performance metric, by default metrics.balanced_accuracy_score
        cv_method : sklearn.model_selection, optional
            valdation strategy, by default model_selection.KFold(n_splits=3, shuffle=False)
        output_name : str, optional
            saving name, by default "LM"
        TRAIN_VAL_SPLIT : bool, optional
            data is split into further validation for early stopping, by default False
        save_coef (boolean):
            if true, save model._coef trained coefficients
        """
        if feature_file is not None:
            self.feature_file = feature_file

        decoder = nm_decode.Decoder(feature_path=self.feature_path,
                                    feature_file=self.feature_file,
                                    model=model,
                                    eval_method=eval_method,
                                    cv_method=cv_method,
                                    threshold_score=True,
                                    TRAIN_VAL_SPLIT=False,
                                    save_coef=False,
                                    get_movement_detection_rate=True,
                                    min_consequent_count=3
                                    )
        decoder.label = self.nm_reader.label
        decoder.target_ch = self.label_name

        if estimate_gridpoints:
            decoder.set_data_grid_points()
            decoder.run_CV_caller("grid_points")
        if estimate_channels:
            decoder.set_data_ind_channels()
            decoder.run_CV_caller("ind_channels")
        if estimate_all_channels_combined:
            if estimate_channels is not True:
                decoder.set_data_ind_channels()
            decoder.run_CV_caller("all_channels_combined")

        decoder.save(output_name)

    def read_results(self, performance_dict=dict(), subject_name=None,
                                 feature_file=None, DEFAULT_PERFORMANCE=0.5, read_grid_points=True,
                                 read_channels=True, read_all_combined=False, ML_model_name='LM',
                                 read_mov_detection_rates=False):
        """Save performances of a given patient into performance_dict

        Parameters
        ----------
        performance_dict : dictionary
            dictionary including decoding performances, by default dictionary
        subject_name : string, optional
            subject name, by default None
        feature_file : string, optional
            feature file, by default None
        DEFAULT_PERFORMANCE : float, optional
            chance performance, by default 0.5
        read_grid_points : bool, optional
            true if grid point performances are read, by default True
        read_channels : bool, optional
            true if channels performances are read, by default True
        read_all_combined : bool, optional
            true if all combined channel performances are read, by default False
        ML_model_name : str, optional
            machine learning model name, by default 'LM'
        read_mov_detection_rates : boolean, by defaulte False
            if True, read movement detection rates, as well as fpr's and tpr's
        Returns
        -------
        dictionary
            performance_dict
        """
        if feature_file is not None:
            self.feature_file = feature_file
        if subject_name is None:
            subject_name = self.feature_file[self.feature_file.find('sub-'):self.feature_file.find('_ses')][4:]

        PATH_ML_ = os.path.join(self.feature_path, self.feature_file,
                                self.feature_file + '_' + ML_model_name + '_ML_RES.p')

        # read ML results
        with open(PATH_ML_, 'rb') as input:
            ML_res = cPickle.load(input)

        performance_dict[subject_name] = {}

        def write_CV_res_in_performance_dict(obj_read, obj_write, read_mov_detection_rates=True):
            obj_write["performance_test"] = np.mean(obj_read["score_test"])
            obj_write["performance_train"] = np.mean(obj_read["score_train"])
            if "coef" in obj_read:
                obj_write["coef"] = np.concatenate(obj_read["coef"]).mean(axis=0)
            if read_mov_detection_rates:
                obj_write["mov_detection_rate_test"] = np.mean(obj_read["mov_detection_rate_test"])
                obj_write["mov_detection_rate_train"] = np.mean(obj_read["mov_detection_rate_train"])
                obj_write["fprate_test"] = np.mean(obj_read["fprate_test"])
                obj_write["fprate_train"] = np.mean(obj_read["fprate_train"])
                obj_write["tprate_test"] = np.mean(obj_read["tprate_test"])
                obj_write["tprate_train"] = np.mean(obj_read["tprate_train"])

        if read_channels:
            ch_to_use = list(np.array(ML_res.settings["ch_names"])
                             [np.where(np.array(ML_res.settings["ch_types"]) == 'ecog')[0]])
            for ch in ch_to_use:

                performance_dict[subject_name][ch] = {}

                if ML_res.settings["sess_right"] is True:
                    cortex_name = "cortex_right"
                else:
                    cortex_name = "cortex_left"

                idx_ = [idx for idx, i in enumerate(ML_res.settings["coord"][cortex_name]["ch_names"])
                        if ch.startswith(i+'-')][0]
                coords = ML_res.settings["coord"][cortex_name]["positions"][idx_]
                performance_dict[subject_name][ch]["coord"] = coords
                write_CV_res_in_performance_dict(ML_res.ch_ind_pr[ch], performance_dict[subject_name][ch],\
                                                 read_mov_detection_rates=True)


        if read_all_combined:
            performance_dict[subject_name]["all_ch_combined"] = {}
            write_CV_res_in_performance_dict(ML_res.all_ch_pr,
                                             performance_dict[subject_name]["all_ch_combined"],\
                                             read_mov_detection_rates=True)

        if read_grid_points:
            performance_dict[subject_name]["active_gridpoints"] = ML_res.active_gridpoints
            for grid_point in range(len(ML_res.settings["grid_cortex"])):
                performance_dict[subject_name]["grid_"+str(grid_point)] = {}
                performance_dict[subject_name]["grid_"+str(grid_point)]["coord"] = \
                    ML_res.settings["grid_cortex"][grid_point]
                if grid_point in ML_res.active_gridpoints:
                    write_CV_res_in_performance_dict(ML_res.gridpoint_ind_pr[grid_point],
                                                     performance_dict[subject_name]["grid_"+str(grid_point)],\
                                                     read_mov_detection_rates=True)
                else:
                    # set non interpolated grid point to default performance
                    performance_dict[subject_name]["grid_"+str(grid_point)]["performance_test"] = DEFAULT_PERFORMANCE
                    performance_dict[subject_name]["grid_"+str(grid_point)]["performance_train"] = DEFAULT_PERFORMANCE
        return performance_dict
