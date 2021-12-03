from abc import ABC, abstractmethod
import pathlib
import numpy as np
import pandas as pd
import os
import json
from enum import Enum

from pyneuromodulation import \
    (nm_projection,
    nm_rereference,
    nm_run_analysis,
    nm_features,
    nm_resample,
    nm_define_nmchannels,
    nm_IO, nm_test_settings)

class GRIDS(Enum):
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

    @abstractmethod
    def __init__(self,
        PATH_SETTINGS=os.path.join(pathlib.Path(__file__).parent.resolve(),\
                                    "nm_settings.json"),
        PATH_NM_CHANNELS:str = str(),
        PATH_OUT:str = os.getcwd(),
        PATH_GRIDS:str = pathlib.Path(__file__).parent.resolve(),
        VERBOSE:bool = False) -> None:

        self.PATH_SETTINGS = PATH_SETTINGS
        self.PATH_NM_CHANNELS = PATH_NM_CHANNELS
        self.PATH_OUT = PATH_OUT
        self.VERBOSE = VERBOSE

        self.settings = nm_IO.read_settings(self.PATH_SETTINGS)

        if True in [self.settings["project_cortex"],
                    self.settings["project_subcortex"]]:
            self.grid_cortex, self.grid_subcortex = self.set_grids(
                self.settings,
                self.PATH_GRIDS,
                GRIDS
                )

    @abstractmethod
    def add_coordinates(self) -> None:
        """This method is implemented differently 
           for BIDS and real time data anylsis
        """
        pass

    @abstractmethod
    def run(self, ieeg_batch: np.array) -> None:
        pass

    def set_run(self):

        self.CH_NAMES_USED, self.CH_TYPES_USED, self.FEATURE_IDX, self.LABEL_IDX = \
            self.set_ch_info(self.nm_channels)

        self.features = self.set_features(self.settings, self.VERBOSE)

        self.resample = self.set_resampling(self.settings)

        self.rereference, self.nm_channels = self.set_rereference(
            self.settings, self.nm_channels
        )

        self. projection = self.set_projection(self.settings)

        self.run_analysis = nm_run_analysis.Run(
            self.features,
            self.settings,
            self.rereference,
            self.projection,
            self.resample,
            self.VERBOSE
        )

    def set_features(self, settings:dict, VERBOSE:bool) -> None:
        """initialize feature class from settings"""
        return nm_features.Features(
            settings,
            VERBOSE
        )

    def set_fs(self, fs: int) -> None:
        self.fs = fs

    def set_rereference(self, settings:dict, nm_channels:pd.DataFrame) -> None:
        if settings["methods"]["re_referencing"] is True:
            rereference = nm_rereference.RT_rereference(
                nm_channels, split_data=False)
        else:
            rereference = None
            # reset nm_channels from default values
            nm_channels["rereference"] = None
            nm_channels["new_name"] = nm_channels["name"]
        return rereference, nm_channels

    def set_resampling(self, settings:dict) -> None:
        if settings["methods"]["raw_resampling"] is True:
            resample = nm_resample.Resample(settings)
        else:
            resample = None
        return resample

    def set_linenoise(self, line_noise: int) -> None:
        self.line_noise = line_noise

    def set_grids(self, settings: dict(), PATH_GRIDS: str, GRID_TYPE: GRIDS):
        if settings["project_cortex"] is True:
            grid_cortex = nm_IO.read_grid(PATH_GRIDS, GRID_TYPE.CORTEX)
        else:
            grid_cortex = None
        if settings["project_subcortex"] is True:
            grid_subcortex = nm_IO.read_grid(PATH_GRIDS, GRID_TYPE.SUBCORTEX)
        else:
            grid_subcortex = None
        return grid_cortex, grid_subcortex

    def set_projection(self, settings:dict):
        if any((settings["methods"]["project_cortex"],
                settings["methods"]["project_subcortex"])):
            projection = nm_projection.Projection(settings)
        else:
            projection = None
        return projection
    def get_ch_info(nm_channels: pd.DataFrame):
        """Get used feature and label info from nm_channels"""

        CH_NAMES_USED = nm_channels[nm_channels["used"] == 1]["new_name"].tolist()
        CH_TYPES_USED = nm_channels[nm_channels["used"] == 1]["type"].tolist()
        
        # used channels for feature estimation
        FEATURE_IDX = np.where(nm_channels["used"] &
                               ~nm_channels["target"])[0].tolist()

        # If multiple targets exist, select only the first
        LABEL_IDX = np.where(nm_channels["target"] == 1)[0]

        return CH_NAMES_USED, CH_TYPES_USED, FEATURE_IDX, LABEL_IDX

    def get_nm_channels(self, PATH_NM_CHANNELS:str, **kwargs) -> None:

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
        self.nm_channels = nm_channels
    
    def set_sess_lat(self, coords):
        if len(coords["cortex_left"]["positions"]) == 0:
            sess_right = True
        elif len(coords["cortex_right"]["positions"]) == 0:
            sess_right = False
        return sess_right