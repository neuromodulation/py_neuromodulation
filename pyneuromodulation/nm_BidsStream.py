import os
from pathlib import Path
from re import VERBOSE
import mne
import types
from mne.io.array.array import RawArray
import numpy as np
import pandas as pd

from pyneuromodulation import nm_IO, nm_define_nmchannels, nm_projection, nm_generator, nm_rereference, \
    nm_run_analysis, nm_features, nm_resample, nm_stream, nm_test_settings


class BidsStream(nm_stream.PNStream):

    PATH_ANNOTATIONS: str
    PATH_GRIDS: str
    PATH_BIDS: str
    PATH_RUN: str
    LIMIT_DATA: bool
    LIMIT_LOW: int
    LIMIT_HIGH: int
    ECOG_ONLY: bool
    raw_arr: mne.io.RawArray
    raw_arr_data: np.array
    annot: mne.Annotations
    VERBOSE: bool = False


    def __init__(self,
        PATH_SETTINGS=...,
        PATH_NM_CHANNELS=...,
        PATH_OUT=...,
        PATH_GRIDS: str = ...,
        VERBOSE: bool = ...,
        PATH_RUN:str = str(),
        PATH_ANNOTATIONS:str = str(),
        PATH_BIDS:str = str(),
        LIMIT_DATA:bool = False,
        LIMIT_LOW:int = 0,
        LIMIT_HIGH:int = 25000,
        ECOG_ONLY:bool = False) -> None:

        super().__init__(PATH_SETTINGS=PATH_SETTINGS,
            PATH_NM_CHANNELS=PATH_NM_CHANNELS,
            PATH_OUT=PATH_OUT,
            PATH_GRIDS=PATH_GRIDS,
            VERBOSE=VERBOSE)

        self.PATH_RUN = PATH_RUN
        self.PATH_ANNOTATIONS = PATH_ANNOTATIONS
        self.PATH_BIDS = PATH_BIDS
        self.ECOG_ONLY = ECOG_ONLY
        self.LIMIT_DATA = LIMIT_DATA
        self.LIMIT_LOW = LIMIT_LOW
        self.LIMIT_HIGH = LIMIT_HIGH

        self.raw_arr, self.raw_arr_data, fs, line_noise = nm_IO.read_BIDS_data(
            self.PATH_RUN, self.PATH_BIDS)

        self.set_fs(fs)

        self.set_linenoise(line_noise)

        self.set_nm_channels(PATH_NM_CHANNELS,
                            ch_names=self.raw_arr.get_channel_types(),
                            bads=self.raw_arr.info["bads"],
                            ECOG_ONLY=ECOG_ONLY
                            )

        if not self.PATH_ANNOTATIONS:
            self.annot, self.annot_data, self.raw_arr = nm_IO.get_annotations(
                                                self.PATH_ANNOTATIONS,
                                                self.PATH_RUN,
                                                self.raw_arr
                                                )
        if self.LIMIT_DATA is True:
            self.raw_arr_data = self.raw_arr_data[:, LIMIT_LOW:LIMIT_HIGH]

        self.coord_list, self.coord_names = \
            self.get_bids_coord_list(self.raw_arr)

        if self.coord_list is None and True in [self.settings["project_cortex"],
                                        self.settings["project_subcortex"]]:
            raise ValueError("no coordinates could be loaded from BIDS Dataset")
        else:
            self.coords = self.add_coordinates(self.coord_names, self.coord_list)
            self.sess_rigth = self.set_sess_lat(self.coords)

        self.gen = nm_generator.ieeg_raw_generator(self.raw_arr_data, self.settings)

        self.set_run()

    def run_bids(self):
        self.run()
        self.add_labels()
        self.save_features()

    def run(self):
        """Overwrites nm_stream abstract run method"""
        while True:
            ieeg_batch = next(self.gen, None)
            if ieeg_batch is not None:
                self.run_analysis.run(ieeg_batch)
            else:
                break

    def add_labels(self):
        """add resampled labels to feature dataframe
        """
        self.df_features = nm_IO.add_labels(
            self.run_analysis.feature_arr, self.settings, self.raw_arr_data)

    def save_features(self):
        """save settings.json, nm_channels.csv and features.csv
           and pickled run_analysis including projections
        """
        self.run_analysis.feature_arr = self.df_features  # here the potential label stream is added
        nm_IO.save_features_and_settings(df_=self.df_features, run_analysis=self.run_analysis,
                                         folder_name=os.path.basename(self.PATH_RUN)[:-5],
                                         settings=self.settings)

    def get_bids_coord_list(self, raw_arr:mne.io.RawArray):
        if raw_arr.get_montage() is not None:
            coord_list = np.array(
                list(
                    dict(
                        raw_arr.get_montage().get_positions()["ch_pos"]
                    ).values()
                )
            ).tolist()
            coord_names = np.array(
                list(
                    dict(
                        raw_arr.get_montage().get_positions()["ch_pos"]
                    ).keys()
                )
            ).tolist()
        else:
            coord_list = None
            coord_names = None

        return coord_list, coord_names

    def add_coordinates(self, coord_names: list, coord_list: np.array):
        """set coordinate information to settings from RawArray
        The set coordinate positions are set as lists, since np.arrays cannot be saved in json
        Parameters
        ----------
        raw_arr : mne.io.RawArray
        PATH_GRIDS : string, optional
            absolute path to grid_cortex.tsv and grid_subcortex.tsv, by default: None
        """

        coords = {}

        for coord_region in [coord_loc+"_"+lat 
                             for coord_loc in ["cortex", "subcortex"] \
                             for lat in ["left", "right"]]:

            coords[coord_region] = {}

            ch_type = "ECOG" if "cortex" in coord_region else "LFP"

            coords[coord_region]["ch_names"] = [
                coord_names[ch_idx]
                for ch_idx, ch in enumerate(coord_list)
                if (coord_list[ch_idx][0] > 0)
                and (ch_type in coord_names[ch_idx])
            ]

            # multiply by 1000 to get m instead of mm
            coords[coord_region]["positions"] = (
                1000 * np.array([
                        ch
                        for ch_idx, ch in enumerate(coord_list)
                        if (coord_list[ch_idx][0] > 0)
                        and (ch_type in coord_names[ch_idx])
                    ]
                )
            )

        return coords
