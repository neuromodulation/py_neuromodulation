"""Module that contains NMStream ABC."""

from abc import ABC, abstractmethod
from pathlib import Path
import pickle

import pandas as pd

from py_neuromodulation.nm_run_analysis import DataProcessor
from py_neuromodulation.nm_settings import NMSettings
from py_neuromodulation.nm_types import _PathLike, FeatureName
from py_neuromodulation import nm_IO, PYNM_DIR


class NMStream(ABC):
    def __init__(
        self,
        sfreq: float,
        nm_channels: pd.DataFrame | _PathLike,
        settings: "NMSettings | _PathLike | None" = None,
        line_noise: float | None = 50,
        sampling_rate_features_hz: float | None = None,
        path_grids: _PathLike | None = None,
        coord_names: list | None = None,
        stream_name: str | None = "example_stream",
        stream_lsl: bool = False,
        coord_list: list | None = None,
        verbose: bool = True,
    ) -> None:
        """Stream initialization

        Parameters
        ----------
        sfreq : float
            sampling frequency of data in Hertz
        nm_channels : pd.DataFrame | _PathLike
            parametrization of channels (see nm_define_channels.py for initialization)
        settings : dict | _PathLike | None, optional
            features settings can be a dictionary or path to the nm_settings.json, by default the py_neuromodulation/nm_settings.json are read
        line_noise : float | None, optional
            line noise, by default 50
        sampling_rate_features_hz : float | None, optional
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
        self.settings: NMSettings = NMSettings.load(settings)

        # If features that use frequency ranges are on, test them against nyquist frequency
        use_freq_ranges: list[FeatureName] = [
            "bandpass_filter",
            "stft",
            "fft",
            "welch",
            "bursts",
            "coherence",
            "nolds",
            "bispectrum",
        ]

        need_nyquist_check = any(
            (f in use_freq_ranges for f in self.settings.features.get_enabled())
        )

        if need_nyquist_check:
            assert all(
                fb.frequency_high_hz < sfreq / 2
                for fb in self.settings.frequency_ranges_hz.values()
            ), (
                "If a feature that uses frequency ranges is selected, "
                "the frequency band ranges need to be smaller than the nyquist frequency.\n"
                f"Got sfreq = {sfreq} and fband ranges:\n {self.settings.frequency_ranges_hz}"
            )

        if sampling_rate_features_hz is not None:
            self.settings.sampling_rate_features_hz = sampling_rate_features_hz

        self.nm_channels = self._load_nm_channels(nm_channels)
        if path_grids is None:
            path_grids = PYNM_DIR
        self.path_grids = path_grids
        self.verbose = verbose
        self.sfreq = sfreq
        self.line_noise = line_noise
        self.coord_names = coord_names
        self.coord_list = coord_list
        self.sess_right = None
        self.projection = None
        self.model = None

        self.data_processor = DataProcessor(
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
    def run(self) -> pd.DataFrame:
        """Reinitialize the stream
        This might be handy in case the nm_channels or nm_settings changed
        """

        self.data_processor = DataProcessor(
            sfreq=self.sfreq,
            settings=self.settings,
            nm_channels=self.nm_channels,
            path_grids=self.path_grids,
            coord_names=self.coord_names,
            coord_list=self.coord_list,
            line_noise=self.line_noise,
            verbose=self.verbose,
        )

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

    def load_model(self, model_name: _PathLike) -> None:
        """Load sklearn model, that utilizes predict"""
        with open(model_name, "rb") as fid:
            self.model = pickle.load(fid)

    def save_after_stream(
        self,
        out_path_root: _PathLike = "",
        folder_name: str = "sub",
        feature_arr: pd.DataFrame | None = None,
        save_csv : bool = False,
    ) -> None:
        """Save features, settings, nm_channels and sidecar after run"""

        out_path_root = Path.cwd() if not out_path_root else Path(out_path_root)

        # create derivate folder_name output folder if doesn't exist
        (out_path_root / folder_name).mkdir(parents=True, exist_ok=True)

        self.PATH_OUT = out_path_root
        self.PATH_OUT_folder_name = folder_name

        self.save_sidecar(out_path_root, folder_name)

        if not save_csv:
            feature_arr = feature_arr.head()

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

    def save_nm_channels(self, out_path_root: _PathLike, folder_name: str) -> None:
        self.data_processor.save_nm_channels(out_path_root, folder_name)

    def save_settings(self, out_path_root: _PathLike, folder_name: str) -> None:
        self.data_processor.save_settings(out_path_root, folder_name)

    def save_sidecar(self, out_path_root: _PathLike, folder_name: str) -> None:
        """Save sidecar incduing fs, coords, sess_right to
        out_path_root and subfolder 'folder_name'"""
        additional_args = {"sess_right": self.sess_right}
        self.data_processor.save_sidecar(out_path_root, folder_name, additional_args)
