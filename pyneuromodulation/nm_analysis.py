import os
from numpy.lib.npyio import save
import pandas as pd
from scipy import io
from matplotlib import pyplot as plt
import numpy as np
import bids
from sklearn import linear_model, base, metrics, model_selection
import _pickle as cPickle
from typing import Optional

from pyneuromodulation import nm_decode, nm_IO, nm_plots

target_filter_str = {
    "CLEAN",
    "SQUARED_ROTAWHEEL"
    "SQUARED_EMG",
    "rota_squared",
    "SQUARED_ROTATION"
}
features_reverse_order_plotting = {"stft", "fft", "bandpass"}

class Feature_Reader:

    feature_dir: str
    feature_list: list[str]
    settings: dict
    sidecar: dict
    fs: int
    line_noise: int
    nm_channels: pd.DataFrame
    feature_arr: pd.DataFrame
    ch_names: list[str]
    ch_names_ECOG: list[str]
    decoder: nm_decode.Decoder=None

    def __init__(
        self,
        feature_dir:str,
        feature_file:str
        ) -> None:
        """Feature_Reader enables analysis methods on top of NM_reader and NM_Decoder

        Parameters
        ----------
        feature_dir : str, optional
            Path to py_neuromodulation estimated feature runs, where each feature is a folder,
        feature_file : str, optional
            specific feature run, if None it is set to the first feature folder in feature_dir
        """
        self.feature_dir = feature_dir
        self.feature_list = nm_IO.get_run_list_indir(self.feature_dir)
        if feature_file is None:
            self.feature_file = self.feature_list[0]
        else:
            self.feature_file = feature_file

        PATH_READ_BASENAME = os.path.join(
                self.feature_dir,
                self.feature_file
        )
        PATH_READ_BASENAME = PATH_READ_BASENAME[:len(PATH_READ_BASENAME)-len(".vhdr")]

        PATH_READ_FILE = os.path.join(PATH_READ_BASENAME,
            self.feature_file[:-len(".vhdr")]
        )

        self.settings = nm_IO.read_settings(PATH_READ_FILE)
        self.sidecar = nm_IO.read_sidecar(PATH_READ_FILE)
        self.fs = self.sidecar["fs"]
        self.line_noise = self.sidecar["line_noise"]
        self.nm_channels = nm_IO.read_nm_channels(PATH_READ_FILE)
        self.feature_arr = nm_IO.read_features(PATH_READ_FILE)

        self.ch_names = self.nm_channels.new_name
        self.used_chs = list(self.nm_channels[(self.nm_channels["target"] == 0) &
                        (self.nm_channels["used"] == 1)]["new_name"])
        self.ch_names_ECOG = [ch_name for ch_name in self.ch_names if "ECOG" in ch_name]

        # init plotter
        self.nmplotter =  nm_plots.NM_Plot()

        self.label_name = self._get_target_ch()
        self.label = self.read_target_ch(self.feature_arr,
            self.label_name,
            binarize=True,
            binarize_th=0.3)

    def _get_target_ch(self) -> str:
        target_names = list(self.nm_channels[self.nm_channels["target"] == 1]["name"])
        target_clean = [
            target_name
            for target_name in target_names
                for filter_str in target_filter_str
                    if filter_str.lower() in target_name.lower()
        ]

        if len(target_clean) == 0:
            target = target_names[0]
        else:
            for target_ in target_clean:
                # try to select contralateral label
                if self.sidecar["sess_right"] is True and "LEFT" in target_:
                    target = target_
                    continue
                elif self.sidecar["sess_right"] is False and "RIGHT" in target_:
                    target = target_
                    continue
                if target_ == target_clean[-1]:
                    target = target_clean[0]  # set label to last element
        return target

    @staticmethod
    def read_target_ch(feature_arr:pd.DataFrame,
        label_name:str,
        binarize:bool = True,
        binarize_th:float = 0.3) -> None:

        label = np.nan_to_num(
                np.array(
                    feature_arr[label_name]
                )
            )
        if binarize:
            label = label > binarize_th
        return label

    @staticmethod
    def filter_features(feature_columns: list,
        ch_name: str=None,
        list_feature_keywords: list[str]=None) -> list:
        """filters read features by ch_name and/or modality

        Parameters
        ----------
        feature_columns : list
            [description]
        ch_name : str, optional
            [description], by default None
        list_feature_keywords : list[str], optional
            list of feature strings that need to be in the columns, by default None

        Returns
        -------
        list
            column list that suffice the ch_name and list_feature_keywords
        """

        if ch_name is not None:
            feature_select = [i for i in list(feature_columns) if ch_name in i]
        else:
            feature_select = feature_columns

        if list_feature_keywords is not None:
            feature_select = [
                f
                for f in feature_select
                    if any(x in f for x in list_feature_keywords)
            ]

            if len([mod for mod in features_reverse_order_plotting if mod in list_feature_keywords])>0:
                # flip list s.t. theta band is lowest in subsequent plot
                feature_select = feature_select[::-1]

        return feature_select

    def set_target_ch(self, ch_name:str) -> None:
        self.label = ch_name

    def plot_cort_projection(self) -> None:
        if self.sidecar["sess_right"]:
            ecog_strip = np.array(
                self.sidecar["coords"]["cortex_right"]["positions"]
            )
        else:
            ecog_strip = np.array(
                self.sidecar["coords"]["cortex_left"]["positions"]
            )
        self.nmplotter.plot_cortex(
            cortex_grid=np.array(self.sidecar["grid_cortex"]),
            ecog_strip=ecog_strip,
            grid_color=np.array(
                self.sidecar["proj_matrix_cortex"]
            ).sum(axis=1)
        )

    def plot_features(self,
        ch_names_ECOG=None,
        list_feature_keywords:list[str]=["stft"],
        epoch_len:int=4,
        threshold:float=0.1):
        """Wrapper that call plot_features_per_channel
        for every given ECoG channel

        Parameters
        ----------
        ch_names_ECOG : list, optional
            list of ECoG channel to plot features for, by default None
        """
        
        if ch_names_ECOG is None:
            ch_names_ECOG = self.ch_names_ECOG
        for ch_name_ECOG in ch_names_ECOG:
            self.plot_target_averaged_channel(
                ch=ch_name_ECOG,
                list_feature_keywords=list_feature_keywords,
                epoch_len=epoch_len,
                threshold=threshold
            )

    def plot_target_averaged_channel(self,
        ch: str=None,
        list_feature_keywords: Optional[list[str]]=None,
        epoch_len: int=4,
        threshold: float=0.1):

        filtered_df = self.feature_arr[
            self.filter_features(self.feature_arr.columns, ch, list_feature_keywords)
        ]

        data = np.expand_dims(
            np.array(filtered_df),
            axis=1
        )

        X_epoch, y_epoch = self.get_epochs(
            data,
            self.label,
            epoch_len=4,
            sfreq=self.settings["sampling_rate_features"],
            threshold=threshold
        )

        nm_plots.plot_epochs_avg(
            X_epoch=X_epoch,
            y_epoch=y_epoch,
            epoch_len=epoch_len,
            sfreq=self.settings["sampling_rate_features"],
            feature_names=list(filtered_df.columns),
            feature_str_add="_".join(list_feature_keywords),
            cut_ch_name_cols=True,
            ch_name=ch,
            label_name=self.label_name,
            normalize_data=True,
            show_plot=True,
            save=True,
            OUT_PATH=self.feature_dir,
            feature_file=self.feature_file
        )

    def plot_subject_grid_ch_performance(
        self,
        subject_name=None,
        performance_dict=None,
        plt_grid=False,
        feature_str_add="performance_allch_allgrid"
    ):
        """plot subject specific performance for individual channeal and optional grid points

        Parameters
        ----------
        subject_name : string, optional
            used subject, by default None
        performance_dict : dict, optional
            [description], by default None
        plt_grid : bool, optional
            True to plot grid performances, by default False
        feature_str_add : string, optional
            figure output_name
        """

        ecog_strip_performance = []
        ecog_coords_strip = []
        cortex_grid = []
        grid_performance = []

        if subject_name is None:
            subject_name = self.feature_file[
                self.feature_file.find('sub-'):self.feature_file.find('_ses')
            ][4:]

        ch_ = list(performance_dict[subject_name].keys())

        for ch in ch_:
            if 'grid' not in ch:
                ecog_coords_strip.append(
                    performance_dict[subject_name][ch]["coord"]
                )
                ecog_strip_performance.append(
                    performance_dict[subject_name][ch]["performance_test"]
                )
            elif plt_grid is True and 'gridcortex_' in ch:
                cortex_grid.append(
                    performance_dict[subject_name][ch]["coord"]
                )
                grid_performance.append(
                    performance_dict[subject_name][ch]["performance_test"]
                )

        if len(ecog_coords_strip) > 0:
            ecog_coords_strip = np.vstack(ecog_coords_strip)

        self.nmplotter.plot_cortex(
            cortex_grid=np.array(self.sidecar["grid_cortex"]) 
                            if "grid_cortex" in self.sidecar else None,
            ecog_strip=ecog_coords_strip if len(ecog_coords_strip)>0 else None,
            grid_color=grid_performance if len(grid_performance)>0 else None,
            strip_color=ecog_strip_performance
                if len(ecog_strip_performance)>0 else None,
            sess_right=self.sidecar["sess_right"],
            save=True,
            OUT_PATH=self.feature_dir,
            feature_file=self.feature_file,
            feature_str_add=feature_str_add,
            show_plot=True
        )

    @staticmethod
    def get_epochs(data, y_, epoch_len, sfreq, threshold=0):
        """Return epoched data.

        Parameters
        ----------
        data : np.ndarray
            array of extracted features of shape (n_samples, n_channels, n_features)
        y_ : np.ndarray
            array of labels e.g. ones for movement and zeros for
            no movement or baseline corr. rotameter data
        sfreq : int/float
            sampling frequency of data
        epoch_len : int
            length of epoch in seconds
        threshold : int/float
            (Optional) threshold to be used for identifying events
            (default=0 for y_tr with only ones
            and zeros)

        Returns
        -------
        epoch_ np.ndarray
            array of epoched ieeg data with shape (epochs,samples,channels,features)
        y_arr np.ndarray
            array of epoched event label data with shape (epochs,samples)
        """

        epoch_lim = int(epoch_len * sfreq)

        ind_mov = np.where(
            np.diff(np.array(y_ > threshold) * 1) == 1
        )[0]

        low_limit = ind_mov > epoch_lim / 2
        up_limit = ind_mov < y_.shape[0] - epoch_lim / 2

        ind_mov = ind_mov[low_limit & up_limit]

        epoch_ = np.zeros([ind_mov.shape[0],
            epoch_lim,
            data.shape[1],data.shape[2]]
        )

        y_arr = np.zeros([ind_mov.shape[0], int(epoch_lim)])

        for idx, i in enumerate(ind_mov):

            epoch_[idx, :, :, :] = data[
                i - epoch_lim // 2:i + epoch_lim // 2, :, :
            ]

            y_arr[idx, :] = y_[i - epoch_lim // 2:i + epoch_lim // 2]

        return epoch_, y_arr

    def set_decoder(self,
        decoder:nm_decode.Decoder = None,
        TRAIN_VAL_SPLIT=False,
        save_coef=False,
        model:base.BaseEstimator = linear_model.LogisticRegression,
        eval_method=metrics.r2_score,
        cv_method:model_selection.BaseCrossValidator=\
            model_selection.KFold(n_splits=3, shuffle=False),
        get_movement_detection_rate:bool=False,
        min_consequent_count=3,
        threshold_score=True
        ):
        if decoder is not None:
            self.decoder = decoder
        else:

            self.decoder = nm_decode.Decoder(
                features=self.feature_arr,
                label=self.label,
                label_name=self.label_name,
                used_chs=self.used_chs,
                model=model,
                eval_method=eval_method,
                cv_method=cv_method,
                threshold_score=threshold_score,
                TRAIN_VAL_SPLIT=TRAIN_VAL_SPLIT,
                save_coef=save_coef,
                get_movement_detection_rate=get_movement_detection_rate,
                min_consequent_count=min_consequent_count
            )

    def run_ML_model(self,
        feature_file:str=None,
        estimate_gridpoints:bool=False,
        estimate_channels:bool=True,
        estimate_all_channels_combined:bool=False,
        output_name:str = "LM",
        save_results:bool = False
        ):
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
        if feature_file is None:
            feature_file = self.feature_file

        if estimate_gridpoints:
            self.decoder.set_data_grid_points()
            _ = self.decoder.run_CV_caller("grid_points")
        if estimate_channels:
            self.decoder.set_data_ind_channels()
            _ = self.decoder.run_CV_caller("ind_channels")
        if estimate_all_channels_combined:
            if estimate_channels is not True:
                self.decoder.set_data_ind_channels()
            _ = self.decoder.run_CV_caller("all_channels_combined")

        if save_results:
            self.decoder.save(
                self.feature_dir,
                self.feature_file[:-len(".vhdr")],
                output_name
            )

        return self.read_results(
            read_grid_points=estimate_gridpoints,
            read_all_combined=estimate_all_channels_combined,
            read_channels=estimate_channels,
            ML_model_name=output_name,
            read_mov_detection_rates=self.decoder.get_movement_detection_rate
        )

    def read_results(self,
        performance_dict: dict = {},
        subject_name: str = None,
        DEFAULT_PERFORMANCE:float = 0.5,
        read_grid_points:bool = True,
        read_channels:bool=True,
        read_all_combined:bool=False,
        ML_model_name:str='LM',
        read_mov_detection_rates:bool=False):
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

        if ".vhdr" in self.feature_file:
            feature_file = self.feature_file[:-len(".vhdr")]
        else:
            feature_file = self.feature_file

        if subject_name is None:
            subject_name = feature_file[feature_file.find('sub-'):feature_file.find('_ses')][4:]

        PATH_ML_ = os.path.join(self.feature_dir, feature_file,
                                feature_file + '_' + ML_model_name + '_ML_RES.p')

        # read ML results
        with open(PATH_ML_, 'rb') as input:
            ML_res = cPickle.load(input)

        performance_dict[subject_name] = {}

        def write_CV_res_in_performance_dict(
            obj_read, obj_write,
            read_mov_detection_rates=read_mov_detection_rates
        ):
            obj_write["performance_test"] = np.mean(obj_read["score_test"])
            obj_write["performance_train"] = np.mean(obj_read["score_train"])
            if "coef" in obj_read:
                obj_write["coef"] = np.concatenate(obj_read["coef"]).mean(axis=0)
            if read_mov_detection_rates:
                obj_write["mov_detection_rate_test"] = np.mean(
                    obj_read["mov_detection_rate_test"]
                )
                obj_write["mov_detection_rate_train"] = np.mean(
                    obj_read["mov_detection_rate_train"]
                )
                obj_write["fprate_test"] = np.mean(obj_read["fprate_test"])
                obj_write["fprate_train"] = np.mean(obj_read["fprate_train"])
                obj_write["tprate_test"] = np.mean(obj_read["tprate_test"])
                obj_write["tprate_train"] = np.mean(obj_read["tprate_train"])

        if read_channels:

            ch_to_use = self.ch_names_ECOG
            for ch in ch_to_use:

                performance_dict[subject_name][ch] = {}

                if self.sidecar["sess_right"] is True:
                    cortex_name = "cortex_right"
                else:
                    cortex_name = "cortex_left"

                idx_ = [idx 
                    for idx, i in enumerate(
                        self.sidecar["coords"][cortex_name]["ch_names"]
                    )
                        if ch.startswith(i+'-')
                ][0]

                coords = self.sidecar["coords"][cortex_name]["positions"][idx_]
                performance_dict[subject_name][ch]["coord"] = coords
                write_CV_res_in_performance_dict(
                    ML_res.ch_ind_results[ch],
                    performance_dict[subject_name][ch],
                    read_mov_detection_rates=read_mov_detection_rates
                )

        if read_all_combined:
            performance_dict[subject_name]["all_ch_combined"] = {}
            write_CV_res_in_performance_dict(
                ML_res.all_ch_results,
                performance_dict[subject_name]["all_ch_combined"],\
                read_mov_detection_rates=read_mov_detection_rates
            )

        if read_grid_points:
            performance_dict[subject_name]["active_gridpoints"] = ML_res.active_gridpoints

            for project_settings, grid_type in zip(
                ["project_cortex", "project_subcortex"],
                ["gridcortex_", "gridsubcortex_"]
            ):
                if self.settings["methods"][project_settings] is False:
                    continue

                # the sidecar keys are grid_cortex and subcortex_grid
                for grid_point in range(
                    len(self.sidecar["grid_"+project_settings.split("_")[1]])
                ):

                    gp_str = grid_type+str(grid_point)

                    performance_dict[subject_name][gp_str] = {}
                    performance_dict[subject_name][gp_str]["coord"] = \
                        self.sidecar[
                            "grid_"+project_settings.split("_")[1]
                        ][grid_point]

                    if gp_str in ML_res.active_gridpoints:
                        write_CV_res_in_performance_dict(
                            ML_res.gridpoint_ind_results[gp_str],
                            performance_dict[subject_name][gp_str],
                            read_mov_detection_rates=read_mov_detection_rates
                        )
                    else:
                        # set non interpolated grid point to default performance
                        performance_dict[subject_name][gp_str]["performance_test"] = \
                            DEFAULT_PERFORMANCE
                        performance_dict[subject_name][gp_str]["performance_train"] = \
                            DEFAULT_PERFORMANCE

        return performance_dict
