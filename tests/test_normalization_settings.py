import os
import unittest

import py_neuromodulation as nm
from py_neuromodulation import nm_define_nmchannels, nm_IO, nm_settings


class TestNormSettings(unittest.TestCase):
    def setUp(self) -> None:
        """
        Load BIDS data, initialize variables for testing different settings.
        """
        SCRIPT_DIR = os.path.join(os.path.abspath("."), "examples")
        sub = "000"
        ses = "right"
        task = "force"
        run = 3
        datatype = "ieeg"

        # Define run name and access paths in the BIDS format.
        self.RUN_NAME = (
            f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_{datatype}.vhdr"
        )

        PATH_RUN = os.path.join(
            (os.path.join(SCRIPT_DIR, "data")),
            f"sub-{sub}",
            f"ses-{ses}",
            datatype,
            self.RUN_NAME,
        )
        PATH_BIDS = os.path.join(SCRIPT_DIR, "data")

        (
            raw,
            self.data,
            self.sfreq,
            self.line_noise,
            self.coord_list,
            self.coord_names,
        ) = nm_IO.read_BIDS_data(
            PATH_RUN=PATH_RUN, BIDS_PATH=PATH_BIDS, datatype=datatype
        )
        self.data = self.data[: int(5 * self.sfreq)]
        self.settings = nm_settings.get_default_settings()

        # Provide a path for the output data. Each re-referencing method has their PATH_OUT
        self.PATH_OUT = os.path.join(
            SCRIPT_DIR, "data", "derivatives", "test_normalization"
        )
        self.nm_channels = nm_define_nmchannels.set_channels(
            ch_names=raw.ch_names,
            ch_types=raw.get_channel_types(),
            reference="default",
            bads=raw.info["bads"],
            new_names="default",
            used_types=("ecog",),  # We focus only on LFP data
            target_keywords=("SQUARED_ROTATION",),
        )

    def test_fast_compute_settings(self) -> None:
        """
        Try if normalization on fast compute settings works.
        No assertion in the end, only want to see if it raises any errors
        """
        self.setUp()
        settings = nm_settings.set_settings_fast_compute(self.settings.copy())
        stream = nm.Stream(
            sfreq=self.sfreq,
            settings=settings,
            line_noise=self.line_noise,
            nm_channels=self.nm_channels,
            coord_names=self.coord_names,
            coord_list=self.coord_list,
            path_grids=None,
            verbose=False,
        )
        stream.run(
            data=self.data,
            out_path_root=self.PATH_OUT,
            folder_name=self.RUN_NAME,
        )

    def test_quantile_norm(self) -> None:
        self.setUp()
        settings = nm_settings.set_settings_fast_compute(self.settings.copy())
        settings["preprocessing"] = ["raw_normalization"]
        settings["postprocessing"]["feature_normalization"] = True
        settings["raw_normalization_settings"][
            "normalization_method"
        ] = "quantile"
        settings["feature_normalization_settings"][
            "normalization_method"
        ] = "quantile"
        stream = nm.Stream(
            sfreq=self.sfreq,
            settings=settings,
            line_noise=self.line_noise,
            nm_channels=self.nm_channels,
            coord_names=self.coord_names,
            coord_list=self.coord_list,
            path_grids=None,
            verbose=False,
        )
        stream.run(
            data=self.data,
            out_path_root=self.PATH_OUT,
            folder_name=self.RUN_NAME,
        )

    def test_zscore_median_norm(self) -> None:        
        self.setUp()
        settings = nm_settings.set_settings_fast_compute(self.settings.copy())
        settings["preprocessing"] = ["raw_normalization"]
        settings["postprocessing"]["feature_normalization"] = True
        settings["raw_normalization_settings"][
            "normalization_method"
        ] = "zscore-median"
        settings["feature_normalization_settings"][
            "normalization_method"
        ] = "zscore-median"
        stream = nm.Stream(
            sfreq=self.sfreq,
            settings=settings,
            line_noise=self.line_noise,
            nm_channels=self.nm_channels,
            coord_names=self.coord_names,
            coord_list=self.coord_list,
            path_grids=None,
            verbose=False,
        )
        stream.run(
            data=self.data,
            out_path_root=self.PATH_OUT,
            folder_name=self.RUN_NAME,
        )

    def test_minmax_norm(self) -> None:
        self.setUp()
        settings = nm_settings.set_settings_fast_compute(self.settings.copy())
        settings["preprocessing"] = ["raw_normalization"]
        settings["postprocessing"]["feature_normalization"] = True
        settings["raw_normalization_settings"][
            "normalization_method"
        ] = "minmax"
        settings["feature_normalization_settings"][
            "normalization_method"
        ] = "minmax"
        stream = nm.Stream(
            sfreq=self.sfreq,
            settings=settings,
            line_noise=self.line_noise,
            nm_channels=self.nm_channels,
            coord_names=self.coord_names,
            coord_list=self.coord_list,
            path_grids=None,
            verbose=False,
        )
        stream.run(
            data=self.data,
            out_path_root=self.PATH_OUT,
            folder_name=self.RUN_NAME,
        )
