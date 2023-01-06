"""Module that contains PNStream ABC."""
from abc import ABC, abstractmethod
import os
import pathlib
import _pickle as cPickle

import pandas as pd
from sklearn import base

from py_neuromodulation import (
    nm_features,
    nm_IO,
    nm_plots,
    nm_run_analysis,
)

_PathLike = str | os.PathLike


class PNStream(ABC):

    settings: dict
    nm_channels: pd.DataFrame
    run_analysis: nm_run_analysis.DataProcessor
    features: nm_features.Features
    coords: dict
    sfreq: int | float
    path_grids: _PathLike | None
    model: base.BaseEstimator | None
    sess_right: bool | None
    verbose: bool

    def __init__(
        self,
        sfreq: int | float,
        nm_channels: pd.DataFrame | _PathLike,
        settings: dict | _PathLike | None = None,
        line_noise: int | float | None = None,
        path_grids: _PathLike | None = None,
        coords: dict | None = None,
        coord_names: list | None = None,
        coord_list: list | None = None,
        verbose: bool = True,
    ) -> None:
        if settings is None:
            settings = (
                pathlib.Path(__file__).parent.resolve() / "nm_settings.json"
            )
        self.settings = self._load_settings(settings)
        self.nm_channels = self._load_nm_channels(nm_channels)
        if path_grids is None:
            path_grids = pathlib.Path(__file__).parent.resolve()
        self.path_grids = path_grids
        self.verbose = verbose
        if coords is None:
            self.coords = {}
        else:
            self.coords = coords
        self.sfreq = sfreq
        self.sess_right = None
        self.projection = None
        self.model = None
        self.run_analysis = nm_run_analysis.DataProcessor(
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
            return nm_IO.load_nm_channels(nm_channels)
        return nm_channels

    @staticmethod
    def _load_settings(settings: dict | _PathLike) -> dict:
        if isinstance(settings, dict):
            return settings
        return nm_IO.read_settings(str(settings))

    def load_model(self, model_name: _PathLike) -> None:
        """Load sklearn model, that utilizes predict"""
        with open(model_name, "rb") as fid:
            self.model = cPickle.load(fid)

    def plot_cortical_projection(self) -> None:
        """plot projection of cortical grid electrodes on cortex"""
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

    def save_settings(
        self, out_path_root: _PathLike, folder_name: str
    ) -> None:
        self.run_analysis.save_settings(out_path_root, folder_name)

    def save_sidecar(self, out_path_root: _PathLike, folder_name: str) -> None:
        """Save sidecar incuding fs, coords, sess_right to
        out_path_root and subfolder 'folder_name'"""
        additional_args = {"sess_right": self.sess_right}
        self.run_analysis.save_sidecar(
            out_path_root, folder_name, additional_args
        )
