"""This module contains the class to process a given batch of data."""

from time import time
from typing import TYPE_CHECKING
import numpy as np

from py_neuromodulation import logger
from py_neuromodulation.utils.types import _PathLike
from py_neuromodulation.features import FeatureProcessors
from py_neuromodulation.utils import io
from py_neuromodulation.stream.settings import NMSettings

if TYPE_CHECKING:
    from py_neuromodulation.processing.projection import Projection
    import pandas as pd


class DataProcessor:
    def __init__(
        self,
        sfreq: float,
        settings: NMSettings | _PathLike,
        channels: "pd.DataFrame | _PathLike",
        coord_names: list | None = None,
        coord_list: list | None = None,
        line_noise: float | None = None,
        path_grids: _PathLike | None = None,
        verbose: bool = True,
    ) -> None:
        from py_neuromodulation.processing import DataPreprocessor

        """Initialize run class.

        Parameters
        ----------
        settings : settings.NMSettings object
        channels : pd.DataFrame | _PathLike
            Initialized pd.DataFrame with channel specific information.
            The path to a channels.csv can be also passed.
        coord_names : list | None
            list of coordinate names
        coord_list : list | None
            list of list of 3D coordinates
        path_grids : _PathLike | None
            List to grid_cortex.tsv and grid_subcortex.tsv for grid point projection
        verbose : boolean
            if True, log signal processed and computation time
        """

        self.settings = NMSettings.load(settings)
        self.channels = io.load_channels(channels)

        self.sfreq_features: float = self.settings.sampling_rate_features_hz
        self._sfreq_raw_orig: float = sfreq
        self.sfreq_raw: float = sfreq // 1
        self.line_noise: float | None = line_noise
        self.path_grids: _PathLike | None = path_grids
        self.verbose: bool = verbose

        self.features_previous = None

        (self.ch_names_used, _, self.feature_idx, _) = self._get_ch_info()

        self.preprocessors = DataPreprocessor(
            settings=self.settings,
            channels=self.channels,
            sfreq=self.sfreq_raw,
            line_noise=self.line_noise,
        )

        if self.settings.postprocessing.feature_normalization:
            from py_neuromodulation.processing.normalization import FeatureNormalizer

            self.feature_normalizer = FeatureNormalizer(self.settings)

        self.features = FeatureProcessors(
            settings=self.settings,
            ch_names=self.ch_names_used,
            sfreq=self.sfreq_raw,
        )

        if coord_list is not None and coord_names is not None:
            self.coords = self._set_coords(
                coord_names=coord_names, coord_list=coord_list
            )

        self.projection = self._get_projection(self.settings, self.channels)

        self.cnt_samples = 0

    @staticmethod
    def _add_coordinates(coord_names: list[str], coord_list: list) -> dict:
        """Write cortical and subcortical coordinate information in joint dictionary

        Parameters
        ----------
        coord_names : list[str]
            list of coordinate names
        coord_list : list
            list of list of 3D coordinates

        Returns
        -------
        dict with (sub)cortex_left and (sub)cortex_right ch_names and positions
        """

        def is_left_coord(val: float, coord_region: str) -> bool:
            if coord_region.split("_")[1] == "left":
                return val < 0
            return val > 0

        coords: dict[str, dict[str, list | np.ndarray]] = {}

        for coord_region in [
            coord_loc + "_" + lat
            for coord_loc in ["cortex", "subcortex"]
            for lat in ["left", "right"]
        ]:
            coords[coord_region] = {}

            ch_type = "ECOG" if "cortex" == coord_region.split("_")[0] else "LFP"

            coords[coord_region]["ch_names"] = [
                coord_name
                for coord_name, ch in zip(coord_names, coord_list)
                if is_left_coord(ch[0], coord_region) and (ch_type in coord_name)
            ]

            # multiply by 1000 to get m instead of mm
            positions = []
            for coord, coord_name in zip(coord_list, coord_names):
                if is_left_coord(coord[0], coord_region) and (ch_type in coord_name):
                    positions.append(coord)
            coords[coord_region]["positions"] = (
                np.array(positions, dtype=np.float64) * 1000
            )

        return coords

    def _get_ch_info(
        self,
    ) -> tuple[list[str], list[str], list[int], np.ndarray]:
        """Get used feature and label info from channels"""
        channels = self.channels
        ch_names_used = channels[channels["used"] == 1]["new_name"].tolist()
        ch_types_used = channels[channels["used"] == 1]["type"].tolist()

        # used channels for feature estimation
        feature_idx = np.where(channels["used"] & ~channels["target"])[0].tolist()

        # If multiple targets exist, select only the first
        label_idx = np.where(channels["target"] == 1)[0]

        return ch_names_used, ch_types_used, feature_idx, label_idx

    @staticmethod
    def _get_grids(
        settings: "NMSettings",
        path_grids: _PathLike | None,
    ) -> "tuple[pd.DataFrame | None, pd.DataFrame | None]":
        """Read settings specified grids

        Parameters
        ----------
        settings : settings.NMSettings object
        path_grids : _PathLike | str

        Returns
        -------
        Tuple
            grid_cortex, grid_subcortex,
            might be None if not specified in settings
        """
        if settings.postprocessing.project_cortex:
            grid_cortex = io.read_grid(path_grids, "cortex")
        else:
            grid_cortex = None
        if settings.postprocessing.project_subcortex:
            grid_subcortex = io.read_grid(path_grids, "subcortex")
        else:
            grid_subcortex = None
        return grid_cortex, grid_subcortex

    def _get_projection(
        self, settings: "NMSettings", channels: "pd.DataFrame"
    ) -> "Projection | None":
        from py_neuromodulation.processing.projection import Projection

        """Return projection of used coordinated and grids"""

        if not any(
            (
                settings.postprocessing.project_cortex,
                settings.postprocessing.project_subcortex,
            )
        ):
            return None

        grid_cortex, grid_subcortex = self._get_grids(self.settings, self.path_grids)
        projection = Projection(
            settings=settings,
            grid_cortex=grid_cortex,
            grid_subcortex=grid_subcortex,
            coords=self.coords,
            channels=channels,
            plot_projection=False,
        )
        return projection

    def _set_coords(
        self, coord_names: list[str] | None, coord_list: list | None
    ) -> dict:
        if not any(
            (
                self.settings.postprocessing.project_cortex,
                self.settings.postprocessing.project_subcortex,
            )
        ):
            return {}

        if any((coord_list is None, coord_names is None)):
            raise ValueError(
                "No coordinates could be loaded. Please provide coord_list and"
                f" coord_names. Got: {coord_list=}, {coord_names=}."
            )

        return self._add_coordinates(
            coord_names=coord_names,
            coord_list=coord_list,  # type: ignore # None case handled above
        )

    def process(self, data: np.ndarray) -> dict[str, float]:
        """Given a new data batch, calculate and return features.

        Parameters
        ----------
        data : np.ndarray
            Current batch of raw data

        Returns
        -------
        pandas Series
            Features calculated from current data
        """
        start_time = time()

        nan_channels = np.isnan(data).any(axis=1)

        data = np.nan_to_num(data)[self.feature_idx, :]

        data = self.preprocessors.process_data(data)

        # calculate features
        features_dict = self.features.estimate_features(data)

        # normalize features
        if self.settings.postprocessing.feature_normalization:
            normed_features = self.feature_normalizer.process(
                np.fromiter(features_dict.values(), dtype=np.float64)
            )
            features_dict = {
                key: normed_features[idx]
                for idx, key in enumerate(features_dict.keys())
            }

        # project features to grid
        if self.projection:
            self.projection.project_features(features_dict)

        # check for all features, where the channel had a NaN, that the feature is also put to NaN
        if nan_channels.sum() > 0:
            # TONI: no need to do this if we store both old and new names for the channels
            new_nan_channels = []
            for ch in list(np.array(self.ch_names_used)[nan_channels]):
                for key in features_dict.keys():
                    if ch in key:
                        new_nan_channels.append(key)

            for ch in new_nan_channels:
                features_dict[ch] = np.nan

        if self.verbose:
            logger.info("Last batch took: %.3f seconds to process", time() - start_time)

        return features_dict

    def save_sidecar(
        self,
        out_dir: _PathLike,
        prefix: str = "",
        additional_args: dict | None = None,
    ) -> None:
        """Save sidecar incuding fs, coords, sess_right to out_dir."""

        sidecar: dict = {
            "original_fs": self._sfreq_raw_orig,
            "final_fs": self.sfreq_raw,
            "sfreq": self.sfreq_features,
        }
        if self.projection:
            sidecar["coords"] = self.projection.coords
            if self.settings.postprocessing.project_cortex:
                sidecar["grid_cortex"] = self.projection.grid_cortex
                sidecar["proj_matrix_cortex"] = self.projection.proj_matrix_cortex
            if self.settings.postprocessing.project_subcortex:
                sidecar["grid_subcortex"] = self.projection.grid_subcortex
                sidecar["proj_matrix_subcortex"] = self.projection.proj_matrix_subcortex
        if additional_args is not None:
            sidecar = sidecar | additional_args

        io.save_sidecar(sidecar, out_dir, prefix)

    def save_settings(self, out_dir: _PathLike, prefix: str = "") -> None:
        self.settings.save(out_dir, prefix)

    def save_channels(self, out_dir: _PathLike, prefix: str) -> None:
        io.save_channels(self.channels, out_dir, prefix)

    def save_features(
        self,
        feature_arr: "pd.DataFrame",
        out_dir: _PathLike = "",
        prefix: str = "",
    ) -> None:
        io.save_features(feature_arr, out_dir, prefix)
