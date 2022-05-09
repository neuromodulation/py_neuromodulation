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
    path_out: _PathLike | None
    path_grids: _PathLike | None
    model: base.BaseEstimator
    verbose: bool

    def __init__(
        self,
        nm_channels: pd.DataFrame | _PathLike,
        settings: dict | _PathLike | None = None,
        path_out: _PathLike | None = None,
        path_grids: _PathLike | None = None,
        verbose: bool = True,
    ) -> None:
        if settings is None:
            settings = (
                pathlib.Path(__file__).parent.resolve() / "nm_settings.json"
            )
        if path_out is None:
            path_out = os.get_cwd()
        if path_grids is None:
            path_grids = pathlib.Path(__file__).parent.resolve()

        self.path_out = path_out
        self.path_grids = path_grids
        self.verbose = verbose
        self.settings = self._get_settings(settings)
        self.nm_channels = self._get_nm_channels(nm_channels)
        self.run_analysis = None
        self.coords = {}
        self.sess_right = None
        self.projection = None
        self.sfreq = None

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

    def _set_run(self):
        """Initialize preprocessing, and feature estimation modules"""
        nm_test_settings.test_settings(self.settings, self.nm_channels)

        (CH_NAMES_USED, _, FEATURE_IDX, _) = self._get_ch_info(
            self.nm_channels
        )

        self.features = self._set_features(
            self.settings,
            CH_NAMES_USED,
            self.sfreq,
            self.verbose,
        )
        self.rereference, self.nm_channels = self._set_rereference(
            self.settings, self.nm_channels
        )

        self.resample = self._set_resampling(
            settings=self.settings,
            sfreq_old=self.sfreq,
        )

        self.notch_filter = self._set_notch_filter(
            settings=self.settings,
            sfreq=self.sfreq
            if self.settings["methods"]["raw_resampling"] is False
            else self.settings["raw_resampling_settings"]["resample_freq_hz"],
        )

        if self.coords:
            self.projection = self._get_projection(
                self.settings, self.nm_channels
            )

        self.run_analysis = nm_run_analysis.Run(
            features=self.features,
            settings=self.settings,
            reference=self.rereference,
            projection=self.projection,
            resample=self.resample,
            notch_filter=self.notch_filter,
            verbose=self.verbose,
            feature_idx=FEATURE_IDX,
        )

    def _set_features(
        self,
        settings: dict,
        CH_NAMES_USED: list,
        fs: int,
        VERBOSE: bool,
    ) -> nm_features.Features:
        """initialize feature class from settings"""
        return nm_features.Features(
            s=settings, ch_names=CH_NAMES_USED, fs=fs, verbose=VERBOSE
        )

    def set_fs(self, fs: int | float) -> None:
        self.sfreq = int(fs)

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
        if settings["methods"]["re_referencing"] is True:
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
        if settings["methods"]["raw_resampling"]:
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
    ) -> nm_filter.NotchFilter | None:
        if settings["methods"]["notch_filter"]:
            kwargs = settings.setdefault("notch_filter_settings", {})
            if "line_noise" not in kwargs:
                kwargs["notch_freqs"] = self.line_noise
            return nm_filter.NotchFilter(sfreq=sfreq, **kwargs)
        return

    def set_linenoise(self, line_noise: int) -> None:
        self.line_noise = line_noise

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

        if settings["methods"]["project_cortex"] is True:
            grid_cortex = nm_IO.read_grid(PATH_GRIDS, GRID_TYPE.CORTEX)
        else:
            grid_cortex = None
        if settings["methods"]["project_subcortex"] is True:
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
                settings["methods"]["project_cortex"],
                settings["methods"]["project_subcortex"],
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

    def save_sidecar(self, folder_name: str):
        """Save sidecar incuding fs, line_noise, coords, sess_right to
        PATH_OUT and subfolder 'folder_name'"""

        sidecar = {
            "original_fs": self.sfreq,
            "fs": self.run_analysis.fs,
            "line_noise": self.line_noise,
            "ch_names": self.features.ch_names,
            "sess_right": self.sess_right,
        }
        if self.projection:
            sidecar["coords"] = (self.projection.coords,)
            if self.settings["methods"]["project_cortex"]:
                sidecar["grid_cortex"] = self.projection.grid_cortex
                sidecar[
                    "proj_matrix_cortex"
                ] = self.projection.proj_matrix_cortex
            if self.settings["methods"]["project_subcortex"]:
                sidecar["grid_subcortex"] = self.projection.grid_subcortex
                sidecar[
                    "proj_matrix_subcortex"
                ] = self.projection.proj_matrix_subcortex

        nm_IO.save_sidecar(sidecar, self.path_out, folder_name)

    def save_settings(self, folder_name: _PathLike):
        nm_IO.save_settings(self.settings, self.path_out, folder_name)

    def save_nm_channels(self, folder_name: _PathLike):
        nm_IO.save_nmchannels(self.nm_channels, self.path_out, folder_name)

    def save_features(self, folder_name: _PathLike, feature_arr: pd.DataFrame):
        nm_IO.save_features(feature_arr, self.path_out, folder_name)

    def save_after_stream(
        self, folder_name: _PathLike, feature_arr: pd.DataFrame | None = None
    ) -> None:
        """Save features, settings, nm_channels and sidecar after run"""

        # create derivate folder_name output folder if doesn't exist
        if os.path.exists(os.path.join(self.path_out, folder_name)) is False:
            os.makedirs(os.path.join(self.path_out, folder_name))

        self.save_sidecar(folder_name)

        if feature_arr:
            self.save_features(folder_name, feature_arr)

        self.save_settings(folder_name)

        self.save_nm_channels(folder_name)

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
                self.settings["methods"]["project_cortex"],
                self.settings["methods"]["project_subcortex"],
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
