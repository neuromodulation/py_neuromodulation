from pathlib import PurePath

import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from scipy.stats import zscore as scipy_zscore

from py_neuromodulation.utils import io
from py_neuromodulation.analysis import plots
from py_neuromodulation.analysis.decode import Decoder
from py_neuromodulation.utils.types import _PathLike
from py_neuromodulation.stream.settings import NMSettings


target_filter_str = {
    "CLEAN",
    "SQUARED_EMG",
    "SQUARED_INTERPOLATED_EMG",
    "SQUARED_ROTAWHEEL",
    "SQUARED_ROTATION" "rota_squared",
}
features_reverse_order_plotting = {"stft", "fft", "bandpass"}


class FeatureReader:
    def __init__(
        self,
        feature_dir: _PathLike,
        feature_file: _PathLike = "",
        binarize_label: bool = True,
    ) -> None:
        """Feature_Reader enables analysis methods on top of NM_reader and NM_Decoder

        Parameters
        ----------
        feature_dir : str, optional
            Path to py_neuromodulation estimated feature runs, where each feature is a folder,
        feature_file : str, optional
            specific feature run, if None it is set to the first feature folder in feature_dir
        binarize_label : bool
            binarize label, by default True

        """
        self.feature_dir = feature_dir
        self.feature_list: list[str] = io.get_run_list_indir(self.feature_dir)
        self.feature_file = feature_file if feature_file else self.feature_list[0]

        FILE_BASENAME = PurePath(self.feature_file).stem
        PATH_READ_FILE = str(PurePath(self.feature_dir, FILE_BASENAME, FILE_BASENAME))

        self.settings = NMSettings.from_file(PATH_READ_FILE)
        self.sidecar = io.read_sidecar(PATH_READ_FILE)
        if self.sidecar["sess_right"] is None:
            if "coords" in self.sidecar:
                if len(self.sidecar["coords"]["cortex_left"]["ch_names"]) > 0:
                    self.sidecar["sess_right"] = False
                if len(self.sidecar["coords"]["cortex_right"]["ch_names"]) > 0:
                    self.sidecar["sess_right"] = True
        self.sfreq = self.sidecar["sfreq"]
        self.channels = io.read_channels(PATH_READ_FILE)
        self.feature_arr = io.read_features(PATH_READ_FILE)

        self.ch_names = self.channels.new_name
        self.used_chs = list(
            self.channels[
                (self.channels["target"] == 0) & (self.channels["used"] == 1)
            ]["new_name"]
        )
        self.ch_names_ECOG = self.channels.query(
            '(type=="ecog") and (used == 1) and (status=="good")'
        ).new_name.to_list()

        # init plotter
        self.nmplotter = plots.NM_Plot()
        if self.channels["target"].sum() > 0:
            self.label_name = self._get_target_ch()
            self.label = self.read_target_ch(
                self.feature_arr,
                self.label_name,
                binarize=binarize_label,
                binarize_th=0.3,
            )

    def _get_target_ch(self) -> str:
        target_names = list(self.channels[self.channels["target"] == 1]["name"])
        target_clean = [
            target_name
            for target_name in target_names
            for filter_str in target_filter_str
            if filter_str.lower() in target_name.lower()
        ]

        if len(target_clean) == 0:
            if "ARTIFACT" not in target_names[0]:
                target = target_names[0]
            elif len(target_names) > 1:
                target = target_names[1]
            else:
                target = target_names[0]
        else:
            for target_ in target_clean:
                # try to select contralateral label
                if self.sidecar["sess_right"] and "LEFT" in target_:
                    target = target_
                    continue
                elif not self.sidecar["sess_right"] and "RIGHT" in target_:
                    target = target_
                    continue
                if target_ == target_clean[-1]:
                    target = target_clean[0]  # set label to last element
        return target

    @staticmethod
    def read_target_ch(
        feature_arr: "pd.DataFrame",
        label_name: str,
        binarize: bool = True,
        binarize_th: float = 0.3,
    ) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        feature_arr : pd.DataFrame
            _description_
        label_name : str
            _description_
        binarize : bool, optional
            _description_, by default True
        binarize_th : float, optional
            _description_, by default 0.3

        Returns
        -------
        _type_
            _description_
        """

        label = np.nan_to_num(np.array(feature_arr[label_name]))
        if binarize:
            label = label > binarize_th
        return label

    @staticmethod
    def filter_features(
        feature_columns: list,
        ch_name: str | None = None,
        list_feature_keywords: list[str] | None = None,
    ) -> list:
        """filters read features by ch_name and/or modality

        Parameters
        ----------
        feature_columns : list
        ch_name : str, optional
        list_feature_keywords : list[str], optional
            list of feature strings that need to be in the columns, by default None

        Returns
        -------
        features : list
            column list that suffice the ch_name and list_feature_keywords
        """

        if ch_name is not None:
            feature_select = [i for i in list(feature_columns) if ch_name in i]
        else:
            feature_select = feature_columns

        if list_feature_keywords is not None:
            feature_select = [
                f for f in feature_select if any(x in f for x in list_feature_keywords)
            ]

            if (
                len(
                    [
                        mod
                        for mod in features_reverse_order_plotting
                        if mod in list_feature_keywords
                    ]
                )
                > 0
            ):
                # flip list s.t. theta band is lowest in subsequent plot
                feature_select = feature_select[::-1]

        return feature_select

    def set_target_ch(self, ch_name: str) -> None:
        self.label_name = ch_name

    def normalize_features(
        self,
    ) -> "pd.DataFrame":
        """Normalize feature_arr feature columns

        Returns:
            pd.DataFrame: z-scored feature_arr
        """
        cols_norm = [c for c in self.feature_arr.columns if "time" not in c]
        feature_arr_norm = scipy_zscore(self.feature_arr[cols_norm], nan_policy="omit")
        feature_arr_norm["time"] = self.feature_arr["time"]
        return feature_arr_norm

    def plot_cort_projection(self) -> None:
        """_summary_"""

        if self.sidecar["sess_right"]:
            ecog_strip = np.array(self.sidecar["coords"]["cortex_right"]["positions"])
        else:
            ecog_strip = np.array(self.sidecar["coords"]["cortex_left"]["positions"])
        self.nmplotter.plot_cortex(
            grid_cortex=np.array(self.sidecar["grid_cortex"])
            if "grid_cortex" in self.sidecar
            else None,
            ecog_strip=ecog_strip,
            grid_color=np.array(self.sidecar["proj_matrix_cortex"]).sum(axis=1)
            if "grid_cortex" in self.sidecar
            else None,
            set_clim=False,
        )

    def plot_target_avg_all_channels(
        self,
        ch_names_ECOG=None,
        list_feature_keywords: list[str] = ["stft"],
        epoch_len: int = 4,
        threshold: float = 0.1,
    ):
        """Wrapper that call plot_features_per_channel
        for every given ECoG channel

        Parameters
        ----------
        ch_names_ECOG : list, optional
            list of ECoG channel to plot features for, by default None
        list_feature_keywords : list[str], optional
            keywords to plot, by default ["stft"]
        epoch_len : int, optional
            epoch length in seconds, by default 4
        threshold : float, optional
            threshold for event detection, by default 0.1
        """

        if ch_names_ECOG is None:
            ch_names_ECOG = self.ch_names_ECOG
        for ch_name_ECOG in ch_names_ECOG:
            self.plot_target_averaged_channel(
                ch=ch_name_ECOG,
                list_feature_keywords=list_feature_keywords,
                epoch_len=epoch_len,
                threshold=threshold,
            )

    def plot_target_averaged_channel(
        self,
        ch: str = "",
        list_feature_keywords: list[str] | None = None,
        features_to_plt: list | None = None,
        epoch_len: int = 4,
        threshold: float = 0.1,
        normalize_data: bool = True,
        show_plot: bool = True,
        title: str = "Movement aligned features",
        ytick_labelsize=None,
        figsize_x: float = 8,
        figsize_y: float = 8,
    ) -> None:
        """_summary_

        Parameters
        ----------
        ch : str, optional
        list_feature_keywords : Optional[list[str]], optional
        features_to_plt : list, optional
        epoch_len : int, optional
        threshold : float, optional
        normalize_data : bool, optional
        show_plot : bool, optional
        title : str, optional
            by default "Movement aligned features"
        ytick_labelsize : _type_, optional
        figsize_x : float, optional
            by default 8
        figsize_y : float, optional
            by default 8
        """

        # TODO: This does not work properly when we have bipolar rereferencing

        if features_to_plt is None:
            filtered_df = self.feature_arr[
                self.filter_features(
                    self.feature_arr.columns, ch, list_feature_keywords
                )[::-1]
            ]
        else:
            filtered_df = self.feature_arr[features_to_plt]

        data = np.expand_dims(np.array(filtered_df), axis=1)

        X_epoch, y_epoch = self.get_epochs(
            data,
            self.label,
            epoch_len=epoch_len,
            sfreq=self.settings.sampling_rate_features_hz,
            threshold=threshold,
        )

        plots.plot_epochs_avg(
            X_epoch=X_epoch,
            y_epoch=y_epoch,
            epoch_len=epoch_len,
            sfreq=self.settings.sampling_rate_features_hz,
            feature_names=list(filtered_df.columns),
            feature_str_add="_".join(list_feature_keywords)
            if list_feature_keywords is not None
            else "all",
            cut_ch_name_cols=True,
            ch_name=ch,
            label_name=self.label_name,
            normalize_data=normalize_data,
            show_plot=show_plot,
            save=True,
            OUT_PATH=self.feature_dir,
            feature_file=self.feature_file,
            str_title=title,
            ytick_labelsize=ytick_labelsize,
            figsize_x=figsize_x,
            figsize_y=figsize_y,
        )

    def plot_all_features(
        self,
        ch_used: str | None = None,
        time_limit_low_s: float | None = None,
        time_limit_high_s: float | None = None,
        normalize: bool = True,
        save: bool = False,
        title="all_feature_plt.pdf",
        ytick_labelsize: int = 10,
        clim_low: float | None = None,
        clim_high: float | None = None,
    ):
        """_summary_

        Parameters
        ----------
        ch_used : str, optional
        time_limit_low_s : float, optional
        time_limit_high_s : float, optional
        normalize : bool, optional
        save : bool, optional
        title : str, optional
            default "all_feature_plt.pdf"
        ytick_labelsize : int, optional
            by default 10
        clim_low : float, optional
            by default None
        clim_high : float, optional
            by default None
        """

        if ch_used is not None:
            col_used = [
                c
                for c in self.feature_arr.columns
                if c.startswith(ch_used) or c == "time" or "LABEL" in c or "MOV" in c
            ]
            df = self.feature_arr[col_used[::-1]]
        else:
            df = self.feature_arr[self.feature_arr.columns[::-1]]

        plots.plot_all_features(
            df=df,
            time_limit_low_s=time_limit_low_s,
            time_limit_high_s=time_limit_high_s,
            normalize=normalize,
            save=save,
            title=title,
            ytick_labelsize=ytick_labelsize,
            feature_file=self.feature_file,
            OUT_PATH=self.feature_dir,
            clim_low=clim_low,
            clim_high=clim_high,
        )

    @staticmethod
    def get_performace_sub_strip(performance_sub: dict, plt_grid: bool = False):
        ecog_strip_performance = []
        ecog_coords_strip = []
        cortex_grid = []
        grid_performance = []

        channels_ = performance_sub.keys()

        for ch in channels_:
            if "grid" not in ch and "combined" not in ch:
                ecog_coords_strip.append(performance_sub[ch]["coord"])
                ecog_strip_performance.append(performance_sub[ch]["performance_test"])
            elif plt_grid and "gridcortex_" in ch:
                cortex_grid.append(performance_sub[ch]["coord"])
                grid_performance.append(performance_sub[ch]["performance_test"])

        if len(ecog_coords_strip) > 0:
            ecog_coords_strip = np.vstack(ecog_coords_strip)

        return (
            ecog_strip_performance,
            ecog_coords_strip,
            cortex_grid,
            grid_performance,
        )

    def plot_across_subject_grd_ch_performance(
        self,
        performance_dict=None,
        plt_grid=False,
        feature_str_add="performance_allch_allgrid",
    ):
        ecog_strip_performance = []
        ecog_coords_strip = []
        grid_performance = []
        for sub in performance_dict.keys():
            (
                ecog_strip_performance_sub,
                ecog_coords_strip_sub,
                _,
                grid_performance_sub,
            ) = self.get_performace_sub_strip(performance_dict[sub], plt_grid=plt_grid)
            ecog_strip_performance.extend(ecog_strip_performance_sub)
            ecog_coords_strip.extend(ecog_coords_strip_sub)
            grid_performance.append(grid_performance_sub)
        grid_performance = list(np.vstack(grid_performance).mean(axis=0))
        coords_all = np.array(ecog_coords_strip)
        coords_all[:, 0] = np.abs(coords_all[:, 0])

        self.nmplotter.plot_cortex(
            grid_cortex=np.array(self.sidecar["grid_cortex"])
            if "grid_cortex" in self.sidecar
            else None,
            ecog_strip=coords_all if len(ecog_coords_strip) > 0 else None,
            grid_color=grid_performance if len(grid_performance) > 0 else None,
            strip_color=np.array(ecog_strip_performance)
            if len(ecog_strip_performance) > 0
            else None,
            sess_right=self.sidecar["sess_right"],
            save=True,
            OUT_PATH=self.feature_dir,
            feature_file=self.feature_file,
            feature_str_add=feature_str_add,
            show_plot=True,
        )

    def plot_subject_grid_ch_performance(
        self,
        subject_name=None,
        performance_dict=None,
        plt_grid=False,
        feature_str_add="performance_allch_allgrid",
    ):
        """plot subject specific performance for individual channeal and optional grid points

        Parameters
        ----------
        subject_name : string, optional
            used subject, by default None
        performance_dict : dict, optional
            by default None
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
                self.feature_file.find("sub-") : self.feature_file.find("_ses")
            ][4:]

        (
            ecog_strip_performance,
            ecog_coords_strip,
            cortex_grid,
            grid_performance,
        ) = self.get_performace_sub_strip(
            performance_dict[subject_name], plt_grid=plt_grid
        )

        self.nmplotter.plot_cortex(
            grid_cortex=np.array(self.sidecar["grid_cortex"])
            if "grid_cortex" in self.sidecar
            else None,
            ecog_strip=ecog_coords_strip if len(ecog_coords_strip) > 0 else None,
            grid_color=grid_performance if len(grid_performance) > 0 else None,
            strip_color=ecog_strip_performance
            if len(ecog_strip_performance) > 0
            else None,
            sess_right=self.sidecar["sess_right"],
            save=True,
            OUT_PATH=self.feature_dir,
            feature_file=self.feature_file,
            feature_str_add=feature_str_add,
            show_plot=True,
        )

    def plot_feature_series_time(
        self,
    ):
        plots.plot_feature_series_time(self.feature_arr)

    def plot_corr_matrix(
        self,
    ):
        return plots.plot_corr_matrix(
            self.feature_arr,
        )

    @staticmethod
    def get_epochs(
        data, y_, epoch_len, sfreq, threshold=0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return epoched data.

        Parameters
        ----------
        data : np.ndarray
            array of extracted features of shape (n_samples, n_channels, n_features)
        y_ : np.ndarray
            array of labels e.g. ones for movement and zeros for
            no movement or baseline corr. rotameter data
        epoch_len : int
            length of epoch in seconds
        sfreq : int/float
            sampling frequency of data
        threshold : int/float
            (Optional) threshold to be used for identifying events
            (default=0 for y_tr with only ones
            and zeros)

        Returns
        -------
        epoch_ : np.ndarray
            array of epoched ieeg data with shape (epochs,samples,channels,features)
        y_arr : np.ndarray
            array of epoched event label data with shape (epochs,samples)
        """

        epoch_lim = int(epoch_len * sfreq)

        ind_mov = np.where(np.diff(np.array(y_ > threshold) * 1) == 1)[0]

        low_limit = ind_mov > epoch_lim / 2
        up_limit = ind_mov < y_.shape[0] - epoch_lim / 2

        ind_mov = ind_mov[low_limit & up_limit]

        epoch_ = np.zeros([ind_mov.shape[0], epoch_lim, data.shape[1], data.shape[2]])

        y_arr = np.zeros([ind_mov.shape[0], int(epoch_lim)])

        for idx, i in enumerate(ind_mov):
            epoch_[idx, :, :, :] = data[i - epoch_lim // 2 : i + epoch_lim // 2, :, :]

            y_arr[idx, :] = y_[i - epoch_lim // 2 : i + epoch_lim // 2]

        return epoch_, y_arr

    def set_decoder(
        self,
        decoder: Decoder | None = None,
        TRAIN_VAL_SPLIT=False,
        RUN_BAY_OPT=False,
        save_coef=False,
        model=LogisticRegression,
        eval_method=r2_score,
        cv_method=KFold(n_splits=3, shuffle=False),
        get_movement_detection_rate: bool = False,
        mov_detection_threshold=0.5,
        min_consequent_count=3,
        threshold_score=True,
        bay_opt_param_space: list = [],
        STACK_FEATURES_N_SAMPLES=False,
        time_stack_n_samples=5,
        use_nested_cv=False,
        VERBOSE=False,
        undersampling=False,
        oversampling=False,
        mrmr_select=False,
        pca=False,
        cca=False,
    ):
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = Decoder(
                features=self.feature_arr,
                label=self.label,
                label_name=self.label_name,
                used_chs=self.used_chs,
                model=model,
                eval_method=eval_method,
                cv_method=cv_method,
                threshold_score=threshold_score,
                TRAIN_VAL_SPLIT=TRAIN_VAL_SPLIT,
                RUN_BAY_OPT=RUN_BAY_OPT,
                save_coef=save_coef,
                get_movement_detection_rate=get_movement_detection_rate,
                min_consequent_count=min_consequent_count,
                mov_detection_threshold=mov_detection_threshold,
                bay_opt_param_space=bay_opt_param_space,
                STACK_FEATURES_N_SAMPLES=STACK_FEATURES_N_SAMPLES,
                time_stack_n_samples=time_stack_n_samples,
                VERBOSE=VERBOSE,
                use_nested_cv=use_nested_cv,
                undersampling=undersampling,
                oversampling=oversampling,
                mrmr_select=mrmr_select,
                sfreq=self.sfreq,
                pca=pca,
                cca=cca,
            )

    def run_ML_model(
        self,
        feature_file: str | None = None,
        estimate_gridpoints: bool = False,
        estimate_channels: bool = True,
        estimate_all_channels_combined: bool = False,
        output_name: str = "LM",
        save_results: bool = True,
    ):
        """machine learning model evaluation for ECoG strip channels and/or grid points

        Parameters
        ----------
        feature_file : string, optional
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
        save_results : boolean
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
            _ = self.decoder.run_CV_caller("all_channels_combined")

        if save_results:
            self.decoder.save(
                self.feature_dir,
                self.feature_file
                if ".vhdr" in self.feature_file
                else self.feature_file,
                output_name,
            )

        return self.read_results(
            read_grid_points=estimate_gridpoints,
            read_all_combined=estimate_all_channels_combined,
            read_channels=estimate_channels,
            ML_model_name=output_name,
            read_mov_detection_rates=self.decoder.get_movement_detection_rate,
            read_bay_opt_params=self.decoder.RUN_BAY_OPT,
            read_mrmr=self.decoder.mrmr_select,
            model_save=self.decoder.model_save,
        )

    def read_results(
        self,
        performance_dict: dict = {},
        subject_name: str | None = None,
        DEFAULT_PERFORMANCE: float = 0.5,
        read_grid_points: bool = True,
        read_channels: bool = True,
        read_all_combined: bool = False,
        ML_model_name: str = "LM",
        read_mov_detection_rates: bool = False,
        read_bay_opt_params: bool = False,
        read_mrmr: bool = False,
        model_save: bool = False,
        save_results: bool = False,
        PATH_OUT: str = "",  # Removed None default, save_general_dict does not handle None anyway
        folder_name: str = "",
        str_add: str = "",
    ):
        """Save performances of a given patient into performance_dict from saved nm_decoder

        Parameters
        ----------
        performance_dict : dictionary
            dictionary including decoding performances, by default dictionary
        subject_name : string, optional
            subject name, by default None
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
        read_bay_opt_params : boolean, by default False
        read_mrmr : boolean, by default False
        model_save : boolean, by default False
        save_results : boolean, by default False
        PATH_OUT : string, by default None
        folder_name : string, by default None
        str_add : string, by default None

        Returns
        -------
        performance_dict : dictionary

        """

        if ".vhdr" in self.feature_file:
            feature_file = self.feature_file[: -len(".vhdr")]
        else:
            feature_file = self.feature_file

        if subject_name is None:
            subject_name = feature_file[
                feature_file.find("sub-") : feature_file.find("_ses")
            ][4:]

        PATH_ML_ = PurePath(
            self.feature_dir,
            feature_file,
            feature_file + "_" + ML_model_name + "_ML_RES.p",
        )

        # read ML results
        with open(PATH_ML_, "rb") as input:
            ML_res = pickle.load(input)
            if self.decoder is None:
                self.decoder = ML_res

        performance_dict[subject_name] = {}

        def write_CV_res_in_performance_dict(
            obj_read,
            obj_write,
            read_mov_detection_rates=read_mov_detection_rates,
            read_bay_opt_params=False,
        ):
            def transform_list_of_dicts_into_dict_of_lists(l_):
                dict_out = {}
                for key_, _ in l_[0].items():
                    key_l = []
                    for dict_ in l_:
                        key_l.append(dict_[key_])
                    dict_out[key_] = key_l
                return dict_out

            def read_ML_performances(
                obj_read, obj_write, set_inner_CV_res: bool = False
            ):
                def set_score(
                    key_set: str = "",
                    key_get: str = "",
                    take_mean: bool = True,
                    val=None,
                ):
                    if set_inner_CV_res:
                        key_set = "InnerCV_" + key_set
                        key_get = "InnerCV_" + key_get
                    if take_mean:
                        val = np.mean(obj_read[key_get])
                    obj_write[key_set] = val

                set_score(
                    key_set="performance_test",
                    key_get="score_test",
                    take_mean=True,
                )
                set_score(
                    key_set="performance_train",
                    key_get="score_train",
                    take_mean=True,
                )

                if "coef" in obj_read:
                    set_score(
                        key_set="coef",
                        key_get="coef",
                        take_mean=False,
                        val=np.concatenate(obj_read["coef"]),
                    )

                if read_mov_detection_rates:
                    set_score(
                        key_set="mov_detection_rates_test",
                        key_get="mov_detection_rates_test",
                        take_mean=True,
                    )
                    set_score(
                        key_set="mov_detection_rates_train",
                        key_get="mov_detection_rates_train",
                        take_mean=True,
                    )
                    set_score(
                        key_set="fprate_test",
                        key_get="fprate_test",
                        take_mean=True,
                    )
                    set_score(
                        key_set="fprate_train",
                        key_get="fprate_train",
                        take_mean=True,
                    )
                    set_score(
                        key_set="tprate_test",
                        key_get="tprate_test",
                        take_mean=True,
                    )
                    set_score(
                        key_set="tprate_train",
                        key_get="tprate_train",
                        take_mean=True,
                    )

                if read_bay_opt_params:
                    # transform dict into keys for json saving
                    dict_to_save = transform_list_of_dicts_into_dict_of_lists(
                        obj_read["best_bay_opt_params"]
                    )
                    set_score(
                        key_set="bay_opt_best_params",
                        take_mean=False,
                        val=dict_to_save,
                    )

                if read_mrmr:
                    # transform dict into keys for json saving

                    set_score(
                        key_set="mrmr_select",
                        take_mean=False,
                        val=obj_read["mrmr_select"],
                    )
                if model_save:
                    set_score(
                        key_set="model_save",
                        take_mean=False,
                        val=obj_read["model_save"],
                    )

            read_ML_performances(obj_read, obj_write)

            if len([key_ for key_ in obj_read.keys() if "InnerCV_" in key_]) > 0:
                read_ML_performances(obj_read, obj_write, set_inner_CV_res=True)

        if read_channels:
            ch_to_use = self.ch_names_ECOG
            ch_to_use = self.decoder.used_chs
            for ch in ch_to_use:
                performance_dict[subject_name][ch] = {}

                if "coords" in self.sidecar:
                    if len(self.sidecar["coords"]) > 0:  # check if coords are empty
                        coords_exist = False
                        for cortex_loc in self.sidecar["coords"].keys():
                            for ch_name_coord_idx, ch_name_coord in enumerate(
                                self.sidecar["coords"][cortex_loc]["ch_names"]
                            ):
                                if ch.startswith(ch_name_coord):
                                    coords = self.sidecar["coords"][cortex_loc][
                                        "positions"
                                    ][ch_name_coord_idx]
                                    coords_exist = (
                                        True  # optimally break out of the two loops...
                                    )
                        if not coords_exist:
                            coords = None
                        performance_dict[subject_name][ch]["coord"] = coords
                write_CV_res_in_performance_dict(
                    ML_res.ch_ind_results[ch],
                    performance_dict[subject_name][ch],
                    read_mov_detection_rates=read_mov_detection_rates,
                    read_bay_opt_params=read_bay_opt_params,
                )

        if read_all_combined:
            performance_dict[subject_name]["all_ch_combined"] = {}
            write_CV_res_in_performance_dict(
                ML_res.all_ch_results,
                performance_dict[subject_name]["all_ch_combined"],
                read_mov_detection_rates=read_mov_detection_rates,
                read_bay_opt_params=read_bay_opt_params,
            )

        if read_grid_points:
            performance_dict[subject_name]["active_gridpoints"] = (
                ML_res.active_gridpoints
            )

            for project_settings, grid_type in zip(
                ["project_cortex", "project_subcortex"],
                ["gridcortex_", "gridsubcortex_"],
            ):
                if not self.settings.postprocessing[project_settings]:
                    continue

                # the sidecar keys are grid_cortex and subcortex_grid
                for grid_point in range(
                    len(self.sidecar["grid_" + project_settings.split("_")[1]])
                ):
                    gp_str = grid_type + str(grid_point)

                    performance_dict[subject_name][gp_str] = {}
                    performance_dict[subject_name][gp_str]["coord"] = self.sidecar[
                        "grid_" + project_settings.split("_")[1]
                    ][grid_point]

                    if gp_str in ML_res.active_gridpoints:
                        write_CV_res_in_performance_dict(
                            ML_res.gridpoint_ind_results[gp_str],
                            performance_dict[subject_name][gp_str],
                            read_mov_detection_rates=read_mov_detection_rates,
                            read_bay_opt_params=read_bay_opt_params,
                        )
                    else:
                        # set non interpolated grid point to default performance
                        performance_dict[subject_name][gp_str]["performance_test"] = (
                            DEFAULT_PERFORMANCE
                        )
                        performance_dict[subject_name][gp_str]["performance_train"] = (
                            DEFAULT_PERFORMANCE
                        )

        if save_results:
            io.save_general_dict(
                dict_=performance_dict,
                path_out=PATH_OUT,
                prefix=folder_name,
                str_add=str_add,
            )
        return performance_dict

    @staticmethod
    def get_dataframe_performances(p: dict) -> "pd.DataFrame":
        performances = []
        for sub in p.keys():
            for ch in p[sub].keys():
                if "active_gridpoints" in ch:
                    continue
                dict_add = p[sub][ch].copy()
                dict_add["sub"] = sub
                dict_add["ch"] = ch

                if "all_ch_" in ch:
                    dict_add["ch_type"] = "all ch combinded"
                elif "gridcortex" in ch:
                    dict_add["ch_type"] = "cortex grid"
                else:
                    dict_add["ch_type"] = "electrode ch"
                performances.append(dict_add)

        return pd.DataFrame(performances)
