"""Module that contains PNStream ABC."""
import os
import pathlib
from abc import ABC, abstractmethod
from enum import Enum

import _pickle as cPickle
import numpy as np
import pandas as pd
from sklearn import base

from py_neuromodulation import (
    nm_features,
    nm_filter,
    nm_IO,
    nm_plots,
    nm_projection,
    nm_rereference,
    nm_resample,
    nm_run_analysis,
    nm_test_settings,
)

_PathLike = str | os.PathLike


class GRIDS(Enum):
    """Definition of possible projection grid types"""

    CORTEX = "cortex"
    SUBCORTEX = "subcortex"


class PNStream(ABC):

    settings: dict | _PathLike
    nm_channels: pd.DataFrame | _PathLike
    run_analysis: nm_run_analysis.Run | None
    rereference: nm_rereference.RT_rereference | None
    resample: nm_resample.Resample | None
    notch_filter: nm_filter.NotchFilter | None
    features: nm_features.Features
    coords: dict
    projection: nm_projection.Projection | None
    sfreq: int | None
    path_grids: _PathLike | None
    model: base.BaseEstimator
    verbose: bool

    def __init__(
        self,
        nm_channels: pd.DataFrame | _PathLike,
        settings: dict | _PathLike | None = None,
        path_grids: _PathLike | None = None,
        coords: dict = {},
        verbose: bool = True,
    ) -> None:
        if settings is None:
            settings = (
                pathlib.Path(__file__).parent.resolve() / "nm_settings.json"
            )
        if path_grids is None:
            path_grids = pathlib.Path(__file__).parent.resolve()

        self.path_grids = path_grids
        self.verbose = verbose
        self.settings = self._get_settings(settings)
        self.nm_channels = self._get_nm_channels(nm_channels)
        self.run_analysis = None
        self.coords = coords
        self.sess_right = None
        self.projection = None
        self.sfreq = None
        self.model = None

    @abstractmethod
    def run(self):
        """In this function data is first acquired iteratively
        1. self.get_data()
        2. data processing is called:
        self.run_analysis.process_data(data) to calculate features
        3. optional postprocessing
        e.g. plotting, ML estimation is done
        """

    @abstractmethod
    def _add_timestamp(
        self, feature_series: pd.Series, idx: int | None = None
    ) -> pd.Series:
        """Add to feature_series "time" keyword
        For Bids specify with fs_features, for real time analysis with current time stamp
        """

    def load_model(self, model_name: _PathLike):
        """Load sklearn model, that utilizes predict"""
        with open(model_name, "rb") as fid:
            self.model = cPickle.load(fid)

    def init_stream(
        self,
        sfreq: int | float,
        line_noise: int | float = None,
        features: nm_features.Features = None,
        rereference: nm_rereference.RT_rereference = None,
        resample: nm_resample.Resample = None,
        notch_filter: nm_filter.NotchFilter = None,
        run_analysis: nm_run_analysis.Run = None,
        coord_names: list | None = None,
        coord_list: list | None = None,
        projection: nm_projection.Projection = None,
    ):
        """Initialize preprocessing, and feature estimation modules"""

        self.sfreq = sfreq

        nm_test_settings.test_settings(self.settings, self.nm_channels)

        (CH_NAMES_USED, _, FEATURE_IDX, _) = self._get_ch_info(self.nm_channels)

        self.features = (
            features
            if features is not None
            else self._set_features(
                self.settings,
                CH_NAMES_USED,
                self.sfreq,
                self.verbose,
            )
        )
        if rereference is None:
            self.rereference, self.nm_channels = self._set_rereference(
                self.settings, self.nm_channels
            )
        else:
            # in this case the self.nm_channels 'new_name' column should be
            # specified by the user to indicate rerefereance
            # in the new channel name
            self.rereference = rereference

        self.resample = (
            resample
            if resample is not None
            else self._set_resampling(
                settings=self.settings,
                sfreq_old=self.sfreq,
            )
        )

        self.notch_filter = (
            notch_filter
            if notch_filter is not None
            else self._set_notch_filter(
                settings=self.settings,
                sfreq=self.sfreq
                if self.settings["preprocessing"]["raw_resampling"] is False
                else self.settings["raw_resampling_settings"][
                    "resample_freq_hz"
                ],
                line_noise=line_noise,
            )
        )

        if coord_list is not None and coord_names is not None:
            self.coords = self._set_coords(
                coord_names=coord_names, coord_list=coord_list
            )

            self.projection = (
                projection
                if projection is not None
                else self._get_projection(self.settings, self.nm_channels)
            )

        self.run_analysis = (
            run_analysis
            if run_analysis is not None
            else nm_run_analysis.Run(
                features=self.features,
                settings=self.settings,
                reference=self.rereference,
                projection=self.projection,
                resample=self.resample,
                notch_filter=self.notch_filter,
                verbose=self.verbose,
                feature_idx=FEATURE_IDX,
            )
        )

    def _set_features(
        self,
        settings: dict,
        CH_NAMES_USED: list,
        sfreq: int,
        VERBOSE: bool,
    ) -> nm_features.Features:
        """initialize feature class from settings"""
        return nm_features.Features(
            s=settings,
            ch_names=CH_NAMES_USED,
            sfreq=sfreq,
        )

    def _set_rereference(
        self, settings: dict, nm_channels: pd.DataFrame
    ) -> tuple[nm_rereference.RT_rereference | None, pd.DataFrame | None]:
        """Initialize nm_rereference and update nm_channels
        nm_channels are updated if no rereferencing is specified

        Parameters
        ----------
        settings : dict
            [description]
        nm_channels : pd.DataFrame
            [description]

        Returns
        -------
        Tuple
            nm_rereference object, updated nm_channels DataFrame
        """
        if settings["preprocessing"]["re_referencing"] is True:
            rereference = nm_rereference.RT_rereference(nm_channels)
        else:
            rereference = None
            # reset nm_channels from default values
            nm_channels["rereference"] = None
            nm_channels["new_name"] = nm_channels["name"]
        return rereference, nm_channels

    def _set_resampling(
        self, settings: dict, sfreq_old: int | float
    ) -> nm_resample.Resample | None:
        """Initialize Resampling

        Parameters
        ----------
        settings : dict
        fs : int

        Returns
        -------
        nm_resample.Resample
        """
        if settings["preprocessing"]["raw_resampling"]:
            resample = nm_resample.Resample(
                sfreq_old=sfreq_old,
                sfreq_new=settings["raw_resampling_settings"][
                    "resample_freq_hz"
                ],
            )
        else:
            resample = None
        return resample

    def _set_notch_filter(
        self,
        settings: dict,
        sfreq: int | float,
        line_noise: int | float,
        trans_bandwidth: int = 15,
        notch_widths: int | np.ndarray | None = 3,
    ) -> nm_filter.NotchFilter | None:
        if settings["preprocessing"]["notch_filter"]:
            return nm_filter.NotchFilter(
                sfreq=sfreq,
                line_noise=line_noise,
                notch_widths=notch_widths,
                trans_bandwidth=trans_bandwidth,
            )
        return

    @staticmethod
    def get_grids(settings: dict, PATH_GRIDS: str, GRID_TYPE: GRIDS) -> tuple:
        """Read settings specified grids

        Parameters
        ----------
        settings : dict
        PATH_GRIDS : str
        GRID_TYPE : GRIDS

        Returns
        -------
        Tuple
            grid_cortex, grid_subcortex,
            might be None if not specified in settings
        """

        if settings["postprocessing"]["project_cortex"] is True:
            grid_cortex = nm_IO.read_grid(PATH_GRIDS, GRID_TYPE.CORTEX)
        else:
            grid_cortex = None
        if settings["postprocessing"]["project_subcortex"] is True:
            grid_subcortex = nm_IO.read_grid(PATH_GRIDS, GRID_TYPE.SUBCORTEX)
        else:
            grid_subcortex = None
        return grid_cortex, grid_subcortex

    def _get_projection(
        self, settings: dict, nm_channels: pd.DataFrame
    ) -> nm_projection.Projection | None:
        """Return projection of used coordinated and grids"""

        if any(
            (
                settings["postprocessing"]["project_cortex"],
                settings["postprocessing"]["project_subcortex"],
            )
        ):
            grid_cortex, grid_subcortex = self.get_grids(
                self.settings, self.path_grids, GRIDS
            )
            projection = nm_projection.Projection(
                settings=settings,
                grid_cortex=grid_cortex,
                grid_subcortex=grid_subcortex,
                coords=self.coords,
                nm_channels=nm_channels,
                plot_projection=False,
            )
            return projection
        return

    @staticmethod
    def _get_ch_info(nm_channels: pd.DataFrame):
        """Get used feature and label info from nm_channels"""

        CH_NAMES_USED = nm_channels[nm_channels["used"] == 1][
            "new_name"
        ].tolist()
        CH_TYPES_USED = nm_channels[nm_channels["used"] == 1]["type"].tolist()

        # used channels for feature estimation
        FEATURE_IDX = np.where(nm_channels["used"] & ~nm_channels["target"])[
            0
        ].tolist()

        # If multiple targets exist, select only the first
        LABEL_IDX = np.where(nm_channels["target"] == 1)[0]

        return CH_NAMES_USED, CH_TYPES_USED, FEATURE_IDX, LABEL_IDX

    @staticmethod
    def _get_settings(settings: dict | _PathLike) -> dict:
        if isinstance(settings, dict):
            return settings
        return nm_IO.read_settings(str(settings))

    @staticmethod
    def _get_nm_channels(
        nm_channels: pd.DataFrame | _PathLike,
    ) -> pd.DataFrame:
        """Read nm_channels from path or specify via BIDS arguments.
        Nexessary parameters are then
        ch_names (list),
        ch_types (list),
        bads (list)
        used_types (list)
        target_keywords (list)
        reference Union[list, str]
        """
        if isinstance(nm_channels, pd.DataFrame):
            return nm_channels
        if nm_channels:
            if not os.path.isfile(nm_channels):
                raise ValueError(
                    "PATH_NM_CHANNELS is not a valid file. Got: "
                    f"{nm_channels}"
                )
            return pd.read_csv(nm_channels)

    @staticmethod
    def _get_sess_lat(coords: dict) -> bool:
        if len(coords["cortex_left"]["positions"]) == 0:
            return True
        elif len(coords["cortex_right"]["positions"]) == 0:
            return False
        raise ValueError(
            "Either cortex_left or cortex_right positions must" " be provided."
        )

    def save_sidecar(self, out_path_root: _PathLike, folder_name: str):
        """Save sidecar incuding fs, coords, sess_right to
        out_path_root and subfolder 'folder_name'"""

        sidecar = {
            "original_fs": self.sfreq,
            "sfreq": self.run_analysis.sfreq,
            "sess_right": self.sess_right,
        }
        if self.projection:
            sidecar["coords"] = self.projection.coords
            if self.settings["postprocessing"]["project_cortex"]:
                sidecar["grid_cortex"] = self.projection.grid_cortex
                sidecar[
                    "proj_matrix_cortex"
                ] = self.projection.proj_matrix_cortex
            if self.settings["postprocessing"]["project_subcortex"]:
                sidecar["grid_subcortex"] = self.projection.grid_subcortex
                sidecar[
                    "proj_matrix_subcortex"
                ] = self.projection.proj_matrix_subcortex

        nm_IO.save_sidecar(sidecar, out_path_root, folder_name)

    def save_settings(self, out_path_root: _PathLike, folder_name: str):
        nm_IO.save_settings(self.settings, out_path_root, folder_name)

    def save_nm_channels(self, out_path_root: _PathLike, folder_name: str):
        nm_IO.save_nmchannels(self.nm_channels, out_path_root, folder_name)

    def save_features(
        self,
        out_path_root: _PathLike,
        folder_name: str,
        feature_arr: pd.DataFrame,
    ):
        nm_IO.save_features(feature_arr, out_path_root, folder_name)

    def save_after_stream(
        self,
        out_path_root: _PathLike = None,
        folder_name: str = "sub",
        feature_arr: pd.DataFrame | None = None,
    ) -> None:
        """Save features, settings, nm_channels and sidecar after run"""

        if out_path_root is None:
            out_path_root = os.get_cwd()
        # create derivate folder_name output folder if doesn't exist
        if os.path.exists(os.path.join(out_path_root, folder_name)) is False:
            os.makedirs(os.path.join(out_path_root, folder_name))

        self.save_sidecar(out_path_root, folder_name)

        if feature_arr is not None:
            self.save_features(out_path_root, folder_name, feature_arr)

        self.save_settings(out_path_root, folder_name)

        self.save_nm_channels(out_path_root, folder_name)

    def plot_cortical_projection(self):
        """plot projection of cortical grid electrodes on cortex"""

        if hasattr(self, "features") is False:
            self._set_run()

        ecog_strip = None
        if self.projection is not None:
            ecog_strip = self.projection.ecog_strip

        grid_cortex = None
        if self.projection is not None:
            grid_cortex = self.projection.grid_cortex

        sess_right = None
        if self.projection is not None:
            sess_right = self.projection.sess_right

        nmplotter = nm_plots.NM_Plot(
            ecog_strip=ecog_strip,
            grid_cortex=grid_cortex,
            sess_right=sess_right,
        )
        nmplotter.plot_cortex(set_clim=False)

    def _set_coords(
        self, coord_names: list | None, coord_list: list | None
    ) -> dict:
        if not any(
            (
                self.settings["postprocessing"]["project_cortex"],
                self.settings["postprocessing"]["project_subcortex"],
            )
        ):
            return {}

        if any((coord_list is None, coord_names is None)):
            raise ValueError(
                "No coordinates could be loaded. Please provide coord_list and"
                f" coord_names. Got: {coord_list=}, {coord_names=}."
            )

        return self._add_coordinates(
            coord_names=coord_names, coord_list=coord_list
        )

    @staticmethod
    def _add_coordinates(coord_names: list, coord_list: list) -> dict:
        """set coordinate information to settings from RawArray
        The set coordinate positions are set as lists,
        since np.arrays cannot be saved in json
        Parameters
        ----------
        raw_arr : mne.io.RawArray
        PATH_GRIDS : string, optional
            absolute path to grid_cortex.tsv and grid_subcortex.tsv, by default: None
        """

        def left_coord(val: int | float, coord_region: str) -> bool:
            if coord_region.split("_")[1] == "left":
                return val < 0
            return val > 0

        coords = {}

        for coord_region in [
            coord_loc + "_" + lat
            for coord_loc in ["cortex", "subcortex"]
            for lat in ["left", "right"]
        ]:

            coords[coord_region] = {}

            ch_type = (
                "ECOG" if "cortex" == coord_region.split("_")[0] else "LFP"
            )

            coords[coord_region]["ch_names"] = [
                coord_name
                for coord_name, ch in zip(coord_names, coord_list)
                if left_coord(ch[0], coord_region) and (ch_type in coord_name)
            ]

            # multiply by 1000 to get m instead of mm
            coords[coord_region]["positions"] = 1000 * np.array(
                [
                    coord
                    for coord, coord_name in zip(coord_list, coord_names)
                    if left_coord(coord[0], coord_region)
                    and (ch_type in coord_name)
                ]
            )

        return coords

    def reset_settings(
        self,
    ) -> None:
        for f in self.settings["features"]:
            self.settings["features"][f] = False
        for f in self.settings["preprocessing"]:
            if f == "preprocessing_order":
                self.settings["preprocessing"][f] = []
            else:
                self.settings["preprocessing"][f] = False
        for f in self.settings["postprocessing"]:
            self.settings["postprocessing"][f] = False

    def set_settings_fast_compute(
        self,
    ) -> None:

        self.reset_settings()
        self.settings["features"]["fft"] = True
        self.settings["preprocessing"]["re_referencing"] = True
        self.settings["preprocessing"]["raw_resampling"] = True
        self.settings["preprocessing"]["notch_filter"] = True
        self.settings["preprocessing"]["raw_normalization"] = False
        self.settings["preprocessing"]["preprocessing_order"] = [
            "raw_resampling",
            "notch_filter",
            "re_referencing",
        ]

        self.settings["postprocessing"]["feature_normalization"] = True
        self.settings["postprocessing"]["project_cortex"] = False
        self.settings["postprocessing"]["project_subcortex"] = False
