from abc import ABC, abstractmethod
import pathlib
from sklearn import base
import numpy as np
import pandas as pd
import os
from enum import Enum

from pyneuromodulation import \
    (nm_projection,
    nm_rereference,
    nm_run_analysis,
    nm_features,
    nm_resample,
    nm_define_nmchannels,
    nm_IO,
    nm_plots)

class GRIDS(Enum):
    """Definition of possible projection grid types"""
    CORTEX="cortex"
    SUBCORTEX="subcortex"

class PNStream(ABC):

    resample: nm_resample.Resample
    features: nm_features.Features
    run_analysis: nm_run_analysis.Run
    rereference: nm_rereference.RT_rereference
    projection: nm_projection.Projection
    settings: dict
    nm_channels: pd.DataFrame
    coords: dict
    fs: float
    line_noise: float
    VERBOSE: bool
    PATH_SETTINGS: str
    PATH_NM_CHANNELS: str = str()
    PATH_OUT: str = str()
    PATH_GRIDS: str = str()
    df_features: pd.DataFrame = pd.DataFrame()
    CH_NAMES_USED: list
    CH_TYPES_USED: list
    FEATURE_IDX: list
    LABEL_IDX: list
    grid_cortex: np.array
    grid_subcortex: np.array
    sess_right: bool
    feature_add: pd.DataFrame
    model: base.BaseEstimator

    @abstractmethod
    def __init__(self,
        PATH_SETTINGS=os.path.join(pathlib.Path(__file__).parent.resolve(),\
                                    "nm_settings.json"),
        PATH_NM_CHANNELS:str = str(),
        PATH_OUT:str = os.getcwd(),
        PATH_GRIDS:str = pathlib.Path(__file__).parent.resolve(),
        VERBOSE:bool = True) -> None:

        self.PATH_SETTINGS = PATH_SETTINGS
        self.PATH_NM_CHANNELS = PATH_NM_CHANNELS
        self.PATH_OUT = PATH_OUT
        self.VERBOSE = VERBOSE

        self.settings = nm_IO.read_settings(self.PATH_SETTINGS)

        if True in [self.settings["methods"]["project_cortex"],
                    self.settings["methods"]["project_subcortex"]]:
            self.grid_cortex, self.grid_subcortex = self.set_grids(
                self.settings,
                self.PATH_GRIDS,
                GRIDS
                )

    @abstractmethod
    def _add_coordinates(self) -> None:
        """This method is implemented differently 
           for BIDS and real time data anylsis
        """
        pass

    @abstractmethod
    def get_data(self) -> np.array:
        pass

    def run(self, predict: bool=False) -> None:
        # Loop
        idx = 0
        while True:
            data = self.get_data()
            if data is None:
                break
            feature_series = self.run_analysis.process_data(data)
            # concatenate data to feature_arr
            if idx == 0:
                self.feature_arr = pd.DataFrame([feature_series])
                idx += 1
            else:
                self.feature_arr = self.feature_arr.append(
                    feature_series, ignore_index=True)

            if predict is True:
                prediction = self.model.predict(feature_series)

    def load_model(self, model: base.BaseEstimator):
        """Load sklearn model, that utilizes predict"""
        pass

    def _set_run(self):

        self.CH_NAMES_USED, self.CH_TYPES_USED, self.FEATURE_IDX, self.LABEL_IDX = \
            self._get_ch_info(self.nm_channels)

        self.features = self._set_features(self.settings,
            self.CH_NAMES_USED,
            self.fs,
            self.line_noise,
            self.VERBOSE
            )

        self.resample = self._set_resampling(self.settings, self.fs)

        self.rereference, self.nm_channels = self._set_rereference(
            self.settings, self.nm_channels
        )

        self.projection = self._set_projection(self.settings)
        if self.projection is not None:
            self.sess_right = self._set_sess_lat(self.coords)
        else:
            self.sess_right = None

        self.run_analysis = nm_run_analysis.Run(
            self.features,
            self.settings,
            self.rereference,
            self.projection,
            self.resample,
            self.nm_channels,
            self.coords,
            self.sess_right,
            self.VERBOSE,
            self.FEATURE_IDX
        )

    def _set_features(self, settings:dict,
        CH_NAMES_USED: list,
        fs: int,
        line_noise: int,
        VERBOSE:bool) -> None:
        """initialize feature class from settings"""
        return nm_features.Features(
            settings,
            CH_NAMES_USED,
            fs,
            line_noise,
            VERBOSE
        )

    def set_fs(self, fs: int) -> None:
        self.fs = fs

    def _set_rereference(self, settings:dict, nm_channels:pd.DataFrame) -> None:
        if settings["methods"]["re_referencing"] is True:
            rereference = nm_rereference.RT_rereference(
                nm_channels, split_data=False)
        else:
            rereference = None
            # reset nm_channels from default values
            nm_channels["rereference"] = None
            nm_channels["new_name"] = nm_channels["name"]
        return rereference, nm_channels

    def _set_resampling(self, settings:dict, fs: int) -> None:
        if settings["methods"]["raw_resampling"] is True:
            resample = nm_resample.Resample(settings, fs)
        else:
            resample = None
        return resample

    def set_linenoise(self, line_noise: int) -> None:
        self.line_noise = line_noise

    def set_grids(self, settings: dict(), PATH_GRIDS: str, GRID_TYPE: GRIDS):
        if settings["methods"]["project_cortex"] is True:
            grid_cortex = nm_IO.read_grid(PATH_GRIDS, GRID_TYPE.CORTEX)
        else:
            grid_cortex = None
        if settings["methods"]["project_subcortex"] is True:
            grid_subcortex = nm_IO.read_grid(PATH_GRIDS, GRID_TYPE.SUBCORTEX)
        else:
            grid_subcortex = None
        return grid_cortex, grid_subcortex

    def _set_projection(self, settings:dict):
        if any((settings["methods"]["project_cortex"],
                settings["methods"]["project_subcortex"])):
            projection = nm_projection.Projection(settings, self.grid_cortex,
                self.grid_subcortex, self.coords, plot_projection=False)
        else:
            projection = None
        return projection

    def _get_ch_info(self, nm_channels: pd.DataFrame):
        """Get used feature and label info from nm_channels"""

        CH_NAMES_USED = nm_channels[nm_channels["used"] == 1]["new_name"].tolist()
        CH_TYPES_USED = nm_channels[nm_channels["used"] == 1]["type"].tolist()

        # used channels for feature estimation
        FEATURE_IDX = np.where(nm_channels["used"] &
                               ~nm_channels["target"])[0].tolist()

        # If multiple targets exist, select only the first
        LABEL_IDX = np.where(nm_channels["target"] == 1)[0]

        return CH_NAMES_USED, CH_TYPES_USED, FEATURE_IDX, LABEL_IDX

    def _get_nm_channels(self, PATH_NM_CHANNELS:str, **kwargs) -> None:

        if PATH_NM_CHANNELS and os.path.isfile(PATH_NM_CHANNELS):
            nm_channels = pd.read_csv(PATH_NM_CHANNELS)
        elif None not in [kwargs.get('ch_names', None),
                        kwargs.get('ch_types', None),
                        kwargs.get('bads', None),
                        kwargs.get('ECOG_ONLY', None)]:

            nm_channels = nm_define_nmchannels.set_channels_by_bids(
                ch_names=kwargs.get('ch_names'),
                ch_types=kwargs.get('ch_types'),
                bads=kwargs.get('bads'),
                ECOG_ONLY=kwargs.get('ECOG_ONLY'))
        return nm_channels

    def _set_sess_lat(self, coords):
        if len(coords["cortex_left"]["positions"]) == 0:
            sess_right = True
        elif len(coords["cortex_right"]["positions"]) == 0:
            sess_right = False
        return sess_right

    def save_sidecar(self, folder_name: str):

        sidecar = {
            "fs" : self.fs,
            "line_noise" : self.line_noise,
            "coords" : self.coords,
            "sess_right" : self.sess_right
        }

        if self.settings["methods"]["project_cortex"]:
            sidecar["grid_cortex"] = self.grid_cortex
            sidecar["proj_matrix_cortex"] = \
                self.projection.proj_matrix_cortex
        if self.settings["methods"]["project_subcortex"]:
            sidecar["grid_subcortex"] = self.grid_subcortex
            sidecar["proj_matrix_subcortex"] = \
                self.projection.proj_matrix_subcortex

        nm_IO.save_sidecar(sidecar, self.PATH_OUT, folder_name)

    def save_settings(self, folder_name: str):
        nm_IO.save_settings(self.settings, self.PATH_OUT, folder_name)

    def save_nm_channels(self, folder_name: str):
        nm_IO.save_nmchannels(self.nm_channels, self.PATH_OUT, folder_name)

    def save_features(self, folder_name: str):
        nm_IO.save_features(self.feature_arr, self.PATH_OUT, folder_name)

    def plot_cortical_projection(self):
        """plot projection of cortical grid electrodes on cortex"""
        nmplotter = nm_plots.NM_Plot(ecog_strip=self.projection.ecog_strip,
                grid_cortex=self.projection.grid_cortex,
                sess_right=self.sess_right)
        nmplotter.plot_cortical_projection()