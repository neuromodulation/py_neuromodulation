"""Module that contains PNStream ABC."""

from abc import ABC, abstractmethod
import os
import pathlib
import pickle

import pandas as pd

from py_neuromodulation import nm_IO
from py_neuromodulation.nm_run_analysis import DataProcessor
from py_neuromodulation.nm_settings import get_default_settings
from py_neuromodulation.nm_types import _PathLike

class PNStream(ABC):

    def __init__(
        self,
        sfreq: int | float,
        nm_channels: pd.DataFrame | _PathLike,
        settings: dict | _PathLike | None = None,
        line_noise: int | float | None = 50,
        sampling_rate_features_hz: int | float | None = None,
        path_grids: _PathLike | None = None,
        coord_names: list | None = None,
        coord_list: list | None = None,
        verbose: bool = True,
    ) -> None:
        """Stream initialization

        Parameters
        ----------
        sfreq : int | float
            sampling frequency of data in Hertz
        nm_channels : pd.DataFrame | _PathLike
            parametrization of channels (see nm_define_channels.py for initialization)
        settings : dict | _PathLike | None, optional
            features settings can be a dictionary or path to the nm_settings.json, by default the py_neuromodulation/nm_settings.json are read
        line_noise : int | float | None, optional
            line noise, by default 50
        sampling_rate_features_hz : int | float | None, optional
            feature sampling rate, by default None
        path_grids : _PathLike | None, optional
            path to grid_cortex.tsv and/or gird_subcortex.tsv, by default Non
        coord_names : list | None, optional
            coordinate name in the form [coord_1_name, coord_2_name, etc], by default None
        coord_list : list | None, optional
            coordinates in the form [[coord_1_x, coord_1_y, coord_1_z], [coord_2_x, coord_2_y, coord_2_z],], by default None
        verbose : bool, optional
            print out stream computation time information, by default True
        """
        self.settings = self._load_settings(settings)

        if sampling_rate_features_hz is not None:
            self.settings["sampling_rate_features_hz"] = (
                sampling_rate_features_hz
            )

        self.nm_channels = self._load_nm_channels(nm_channels)
        if path_grids is None:
            path_grids = nm_IO.PYNM_DIR
        self.path_grids = path_grids
        self.verbose = verbose
        self.sfreq = sfreq
        self.line_noise = line_noise
        self.coord_names = coord_names
        self.coord_list = coord_list
        self.sess_right = None
        self.projection = None
        self.model = None

        self.run_analysis = DataProcessor(
            sfreq=self.sfreq,
            settings=self.settings,
            nm_channels=self.nm_channels,
            path_grids=self.path_grids,
            coord_names=coord_names,
            coord_list=coord_list,
            line_noise=line_noise,
            verbose=self.verbose,
        )

    @abstractmethod
    def run(self):
        """Reinitialize the stream
        This might be handy in case the nm_channels or nm_settings changed
        """

        self.run_analysis = DataProcessor(
            sfreq=self.sfreq,
            settings=self.settings,
            nm_channels=self.nm_channels,
            path_grids=self.path_grids,
            coord_names=self.coord_names,
            coord_list=self.coord_list,
            line_noise=self.line_noise,
            verbose=self.verbose,
        )

    @abstractmethod
    def _add_timestamp(
        self, feature_series: pd.Series, cnt_samples: int
    ) -> pd.Series:
        """Add to feature_series "time" keyword
        For Bids specify with fs_features, for real time analysis with current time stamp
        """

    @staticmethod
    def _get_sess_lat(coords: dict) -> bool:
        if len(coords["cortex_left"]["positions"]) == 0:
            return True
        if len(coords["cortex_right"]["positions"]) == 0:
            return False
        raise ValueError(
            "Either cortex_left or cortex_right positions must be provided."
        )

    @staticmethod
    def _load_nm_channels(
        nm_channels: pd.DataFrame | _PathLike,
    ) -> pd.DataFrame:
        if not isinstance(nm_channels, pd.DataFrame):
            nm_channels = nm_IO.load_nm_channels(nm_channels)

        if nm_channels.query("used == 1 and target == 0").shape[0] == 0:
            raise ValueError(
                "No channels selected for analysis that have column 'used' = 1 and 'target' = 0. Please check your nm_channels"
            )

        return nm_channels

    @staticmethod
    def _load_settings(settings: dict | _PathLike | None) -> dict:
        if isinstance(settings, dict):
            return settings
        if settings is None:
            return get_default_settings()
        return nm_IO.read_settings(str(settings))

    def load_model(self, model_name: _PathLike) -> None:
        """Load sklearn model, that utilizes predict"""
        with open(model_name, "rb") as fid:
            self.model = pickle.load(fid)

    def save_after_stream(
        self,
        out_path_root: _PathLike | None = None,
        folder_name: str = "sub",
        feature_arr: pd.DataFrame | None = None,
    ) -> None:
        """Save features, settings, nm_channels and sidecar after run"""

        if out_path_root is None:
            out_path_root = os.getcwd()
        # create derivate folder_name output folder if doesn't exist
        if os.path.exists(os.path.join(out_path_root, folder_name)) is False:
            os.makedirs(os.path.join(out_path_root, folder_name))

        self.PATH_OUT = out_path_root
        self.PATH_OUT_folder_name = folder_name
        self.save_sidecar(out_path_root, folder_name)

        if feature_arr is not None:
            self.save_features(out_path_root, folder_name, feature_arr)

        self.save_settings(out_path_root, folder_name)

        self.save_nm_channels(out_path_root, folder_name)

    def save_features(
        self,
        out_path_root: _PathLike,
        folder_name: str,
        feature_arr: pd.DataFrame,
    ) -> None:
        nm_IO.save_features(feature_arr, out_path_root, folder_name)

    def save_nm_channels(
        self, out_path_root: _PathLike, folder_name: str
    ) -> None:
        self.run_analysis.save_nm_channels(out_path_root, folder_name)

    def save_settings(self, out_path_root: _PathLike, folder_name: str) -> None:
        self.run_analysis.save_settings(out_path_root, folder_name)

    def save_sidecar(self, out_path_root: _PathLike, folder_name: str) -> None:
        """Save sidecar incduing fs, coords, sess_right to
        out_path_root and subfolder 'folder_name'"""
        additional_args = {"sess_right": self.sess_right}
        self.run_analysis.save_sidecar(
            out_path_root, folder_name, additional_args
        )
