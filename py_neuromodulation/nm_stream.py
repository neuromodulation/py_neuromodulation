from abc import ABC, abstractmethod
import pathlib
from sklearn import base
import _pickle as cPickle
import numpy as np
import pandas as pd
import os
from enum import Enum
from typing import Tuple, Union

from py_neuromodulation import (
    nm_projection,
    nm_rereference,
    nm_run_analysis,
    nm_features,
    nm_resample,
    nm_define_nmchannels,
    nm_IO,
    nm_plots,
    nm_test_settings,
)
from py_neuromodulation import nm_notch_filter


class GRIDS(Enum):
    """Definition of possible projection grid types"""

    CORTEX = "cortex"
    SUBCORTEX = "subcortex"


class PNStream(ABC):

    resample: nm_resample.Resample
    features: nm_features.Features
    run_analysis: nm_run_analysis.Run
    rereference: nm_rereference.RT_rereference
    notch_filter: nm_notch_filter.NotchFilter
    projection: nm_projection.Projection
    settings: dict
    nm_channels: pd.DataFrame
    coords: dict = {}
    fs: Union[int, float]
    line_noise: float
    VERBOSE: bool
    PATH_SETTINGS: str
    PATH_NM_CHANNELS: str = str()
    PATH_OUT: str = str()
    PATH_GRIDS: str = str()
    feature_arr: pd.DataFrame = pd.DataFrame()
    CH_NAMES_USED: list
    CH_TYPES_USED: list
    FEATURE_IDX: list
    LABEL_IDX: list
    grid_cortex: np.array
    grid_subcortex: np.array
    sess_right: bool = None
    feature_add: pd.DataFrame
    model: base.BaseEstimator

    @abstractmethod
    def __init__(
        self,
        PATH_SETTINGS=os.path.join(
            pathlib.Path(__file__).parent.resolve(), "nm_settings.json"
        ),
        PATH_NM_CHANNELS: str = str(),
        PATH_OUT: str = os.getcwd(),
        PATH_GRIDS: str = pathlib.Path(__file__).parent.resolve(),
        VERBOSE: bool = True,
    ) -> None:

        self.PATH_SETTINGS = PATH_SETTINGS
        self.PATH_NM_CHANNELS = PATH_NM_CHANNELS
        self.PATH_OUT = PATH_OUT
        self.VERBOSE = VERBOSE

        self.settings = nm_IO.read_settings(self.PATH_SETTINGS)

        if True in [
            self.settings["methods"]["project_cortex"],
            self.settings["methods"]["project_subcortex"],
        ]:
            self.grid_cortex, self.grid_subcortex = self.get_grids(
                self.settings, self.PATH_GRIDS, GRIDS
            )

    @abstractmethod
    def _add_coordinates(self) -> None:
        """Write self.coords either from bids or from separate file
        This method is implemented differently
        for BIDS and real time data anylsis
        """
        pass

    @abstractmethod
    def get_data(
        self,
    ) -> np.array:
        """Get new data batch from acquisition device or from BIDS"""
        pass

    @abstractmethod
    def run(
        self,
    ):
        """In this function data is first acquied iteratively
        1. self.get_data()
        2. data processing is called:
        self.run_analysis.process_data(data) to calculate features
        3. optional postprocessing
        e.g. plotting, ML estimation is done
        """
        pass

    @abstractmethod
    def _add_timestamp(self, feature_series: pd.Series, idx: int = None) -> pd.Series:
        """Add to feature_series "time" keyword
        For Bids specify with fs_features, for real time analysis with current time stamp
        """
        pass

    def load_model(self, model_name: str):
        """Load sklearn model, that utilizes predict"""
        with open(model_name, "rb") as fid:
            self.model = cPickle.load(fid)

    def _set_run(self):
        """Initialize preprocessing, and feature estimation modules"""

        nm_test_settings.test_settings(self.settings, self.nm_channels)

        (
            self.CH_NAMES_USED,
            self.CH_TYPES_USED,
            self.FEATURE_IDX,
            self.LABEL_IDX,
        ) = self._get_ch_info(self.nm_channels)

        self.features = self._set_features(
            self.settings, self.CH_NAMES_USED, self.fs, self.line_noise, self.VERBOSE
        )

        self.resample = self._set_resampling(self.settings, self.fs)

        self.rereference, self.nm_channels = self._set_rereference(
            self.settings, self.nm_channels
        )

        self.notch_filter = self._set_notch_filter(
            settings=self.settings,
            fs=self.fs
            if self.settings["methods"]["raw_resampling"] is False
            else self.settings["raw_resampling_settings"]["resample_freq_hz"],
            line_noise=self.line_noise,
        )

        self.projection = self._get_projection(self.settings, self.nm_channels)
        if "cortex_left" in self.coords or "cortex_right" in self.coords:
            if (
                self.projection is not None
                or len(self.coords["cortex_left"]["positions"])
                or len(self.coords["cortex_right"]["positions"])
            ):
                self.sess_right = self._get_sess_lat(self.coords)
            else:
                self.sess_right = None
        else:
            self.sess_right = None

        self.run_analysis = nm_run_analysis.Run(
            features=self.features,
            settings=self.settings,
            reference=self.rereference,
            projection=self.projection,
            resample=self.resample,
            notch_filter=self.notch_filter,
            verbose=self.VERBOSE,
            feature_idx=self.FEATURE_IDX,
        )

    def _set_features(
        self,
        settings: dict,
        CH_NAMES_USED: list,
        fs: Union[int, float],
        line_noise: int,
        VERBOSE: bool,
    ) -> None:
        """initialize feature class from settings"""
        return nm_features.Features(settings, CH_NAMES_USED, fs, line_noise, VERBOSE)

    def set_fs(self, fs: Union[int, float]) -> None:
        self.fs = fs

    def _set_rereference(
        self, settings: dict, nm_channels: pd.DataFrame
    ) -> tuple[nm_rereference.RT_rereference, pd.DataFrame]:
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
        self, settings: dict, fs: Union[int, float]
    ) -> nm_resample.Resample:
        """Initialize Resampling

        Parameters
        ----------
        settings : dict
        fs : int

        Returns
        -------
        nm_resample.Resample
        """
        if settings["methods"]["raw_resampling"] is True:
            resample = nm_resample.Resample(settings, fs)
        else:
            resample = None
        return resample

    def _set_notch_filter(
        self,
        settings: dict,
        fs: Union[int, float],
        line_noise: int,
        notch_widths: int = 3,
        trans_bandwidth: int = 15,
    ) -> nm_notch_filter.NotchFilter:

        if settings["methods"]["notch_filter"] is True:
            notch_filter = nm_notch_filter.NotchFilter(
                fs=fs,
                line_noise=line_noise,
                notch_widths=notch_widths,
                trans_bandwidth=trans_bandwidth,
            )
        else:
            notch_filter = None
        return notch_filter

    def set_linenoise(self, line_noise: int) -> None:
        self.line_noise = line_noise

    @staticmethod
    def get_grids(settings: dict(), PATH_GRIDS: str, GRID_TYPE: GRIDS) -> Tuple:
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
    ) -> nm_projection.Projection:
        """Return projection of used coordinated and grids"""

        if any(
            (
                settings["methods"]["project_cortex"],
                settings["methods"]["project_subcortex"],
            )
        ):
            projection = nm_projection.Projection(
                settings=settings,
                grid_cortex=self.grid_cortex,
                grid_subcortex=self.grid_subcortex,
                coords=self.coords,
                nm_channels=nm_channels,
                plot_projection=False,
            )
        else:
            projection = None
        return projection

    @staticmethod
    def _get_ch_info(nm_channels: pd.DataFrame):
        """Get used feature and label info from nm_channels"""

        CH_NAMES_USED = nm_channels[nm_channels["used"] == 1]["new_name"].tolist()
        CH_TYPES_USED = nm_channels[nm_channels["used"] == 1]["type"].tolist()

        # used channels for feature estimation
        FEATURE_IDX = np.where(nm_channels["used"] & ~nm_channels["target"])[0].tolist()

        # If multiple targets exist, select only the first
        LABEL_IDX = np.where(nm_channels["target"] == 1)[0]

        return CH_NAMES_USED, CH_TYPES_USED, FEATURE_IDX, LABEL_IDX

    @staticmethod
    def _get_nm_channels(PATH_NM_CHANNELS: str, **kwargs) -> None:
        """Read nm_channels from path or specify via BIDS arguments.
        Nexessary parameters are then
        ch_names (list),
        ch_types (list),
        bads (list)
        used_types (list)
        target_keywords (list)
        reference Union[list, str]
        """

        if PATH_NM_CHANNELS and os.path.isfile(PATH_NM_CHANNELS):
            nm_channels = pd.read_csv(PATH_NM_CHANNELS)
        elif (
            len(
                [
                    1
                    for l in [
                        kwargs.get("ch_names", None),
                        kwargs.get("ch_types", None),
                    ]
                    if len(l) == 0
                ]
            )
            == 0
        ):

            nm_channels = nm_define_nmchannels.set_channels(
                ch_names=kwargs.get("ch_names"),
                ch_types=kwargs.get("ch_types"),
                bads=kwargs.get("bads"),
                used_types=kwargs.get("used_types"),
                target_keywords=kwargs.get("target_keywords"),
                reference=kwargs.get("reference"),
            )
        return nm_channels

    @staticmethod
    def _get_sess_lat(coords):
        if len(coords["cortex_left"]["positions"]) == 0:
            sess_right = True
        elif len(coords["cortex_right"]["positions"]) == 0:
            sess_right = False
        return sess_right

    def save_sidecar(self, folder_name: str):
        """Save sidecar incuding fs, line_noise, coords, sess_right to
        PATH_OUT and subfolder 'folder_name'"""

        sidecar = {
            "original_fs": self.fs,
            "fs": self.run_analysis.fs,
            "line_noise": self.line_noise,
            "ch_names": self.features.ch_names,
            "coords": self.coords,
            "sess_right": self.sess_right,
        }

        if self.settings["methods"]["project_cortex"]:
            sidecar["grid_cortex"] = self.grid_cortex
            sidecar["proj_matrix_cortex"] = self.projection.proj_matrix_cortex
        if self.settings["methods"]["project_subcortex"]:
            sidecar["grid_subcortex"] = self.grid_subcortex
            sidecar["proj_matrix_subcortex"] = self.projection.proj_matrix_subcortex

        nm_IO.save_sidecar(sidecar, self.PATH_OUT, folder_name)

    def save_settings(self, folder_name: str):
        nm_IO.save_settings(self.settings, self.PATH_OUT, folder_name)

    def save_nm_channels(self, folder_name: str):
        nm_IO.save_nmchannels(self.nm_channels, self.PATH_OUT, folder_name)

    def save_features(self, folder_name: str):
        nm_IO.save_features(self.feature_arr, self.PATH_OUT, folder_name)

    def save_after_stream(self, folder_name: str, save_features: bool = True) -> None:
        """Save features, settings, nm_channels and sidecar after run"""

        # create derivate folder_name output folder if doesn't exist
        if os.path.exists(os.path.join(self.PATH_OUT, folder_name)) is False:
            os.makedirs(os.path.join(self.PATH_OUT, folder_name))

        self.save_sidecar(folder_name)
        if save_features is True:
            self.save_features(folder_name)

        self.save_settings(folder_name)

        self.save_nm_channels(folder_name)

    def plot_cortical_projection(self):
        """plot projection of cortical grid electrodes on cortex"""

        if hasattr(self, "features") is False:
            self._set_run()

        nmplotter = nm_plots.NM_Plot(
            ecog_strip=self.projection.ecog_strip
            if self.projection is not None
            else None,
            grid_cortex=self.projection.grid_cortex
            if self.projection is not None
            else None,
            sess_right=self.sess_right,
        )
        nmplotter.plot_cortex(set_clim=False)
