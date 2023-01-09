import unittest
import os
import pdb
import py_neuromodulation as nm
from py_neuromodulation import (
    nm_define_nmchannels,
    nm_IO,
    nm_settings,
    nm_stream_offline
)


class TestNormSettings(unittest.TestCase):
    def set_up(self):
        """
        Load BIDS data, create stream.
        Returns necessary variables for testing different settings.
        :return: stream, sfreq, line_noise, coord_list, coord_names, data, PATH_OUT, RUN_NAME
        """

        RUN_NAME, PATH_RUN, PATH_BIDS, PATH_OUT, datatype = nm_IO.get_paths_example_data()

        (
            raw,
            data,
            sfreq,
            line_noise,
            coord_list,
            coord_names,
        ) = nm_IO.read_BIDS_data(
            PATH_RUN=PATH_RUN, BIDS_PATH=PATH_BIDS, datatype=datatype
        )

        nm_channels = nm_define_nmchannels.set_channels(
            ch_names=raw.ch_names,
            ch_types=raw.get_channel_types(),
            reference="default",
            bads=raw.info["bads"],
            new_names="default",
            used_types=("ecog", "dbs", "seeg"),
            target_keywords=("MOV_RIGHT_CLEAN",),
        )

        settings = nm_settings.get_default_settings()
        settings = nm_settings.reset_settings(settings)

        stream = nm_stream_offline.Stream(
            settings=settings,
            nm_channels=nm_channels,
            path_grids=None,
            verbose=True,
            sfreq=sfreq,
            line_noise=line_noise,
            coord_list=coord_list,
            coord_names=coord_names
        )

        return data, stream

    def test_fast_compute_settings(self):
        """
        Try if normalization on fast compute settings works.
        No assertion in the end, only want to see if it raises any errors
        """
        data, stream = self.set_up()

        stream.set_settings_fast_compute()

        stream.run(
            data=data,
            out_path_root=PATH_OUT,
            folder_name=RUN_NAME,
        )

    def test_quantile_norm(self):
        (
            stream,
            sfreq,
            line_noise,
            coord_list,
            coord_names,
            data,
            nm_channels,
            PATH_OUT,
            RUN_NAME,
        ) = self.set_up()

        stream.set_settings_fast_compute()

        stream.settings["preprocessing"]["raw_normalization"] = True
        stream.settings["preprocessing"]["preprocessing_order"] = [
            "raw_normalization",
        ]
        stream.settings["postprocessing"]["feature_normalization"] = True
        stream.settings["raw_normalization_settings"][
            "normalization_method"
        ] = {
            "mean": False,
            "median": False,
            "zscore": False,
            "zscore-median": False,
            "quantile": True,
            "power": False,
            "robust": False,
            "minmax": False,
        }
        stream.settings["feature_normalization_settings"][
            "normalization_method"
        ] = {
            "mean": False,
            "median": False,
            "zscore": False,
            "zscore-median": False,
            "quantile": True,
            "power": False,
            "robust": False,
            "minmax": False,
        }

        stream.init_stream(
            sfreq=sfreq,
            line_noise=line_noise,
            coord_list=coord_list,
            coord_names=coord_names,
        )

        stream.run(
            data=data,
            out_path_root=PATH_OUT,
            folder_name=RUN_NAME,
        )

    def test_zscore_median_norm(self):
        (
            stream,
            sfreq,
            line_noise,
            coord_list,
            coord_names,
            data,
            nm_channels,
            PATH_OUT,
            RUN_NAME,
        ) = self.set_up()

        stream.set_settings_fast_compute()

        stream.settings["preprocessing"]["raw_normalization"] = True
        stream.settings["preprocessing"]["preprocessing_order"] = [
            "raw_normalization",
        ]
        stream.settings["postprocessing"]["feature_normalization"] = True
        stream.settings["raw_normalization_settings"][
            "normalization_method"
        ] = {
            "mean": False,
            "median": False,
            "zscore": False,
            "zscore-median": True,
            "quantile": False,
            "power": False,
            "robust": False,
            "minmax": False,
        }
        stream.settings["feature_normalization_settings"][
            "normalization_method"
        ] = {
            "zscore-median": True,
        }

        stream.init_stream(
            sfreq=sfreq,
            line_noise=line_noise,
            coord_list=coord_list,
            coord_names=coord_names,
        )

        stream.run(
            data=data,
            out_path_root=PATH_OUT,
            folder_name=RUN_NAME,
        )

    def test_minmax_norm(self):
        (
            stream,
            sfreq,
            line_noise,
            coord_list,
            coord_names,
            data,
            nm_channels,
            PATH_OUT,
            RUN_NAME,
        ) = self.set_up()

        stream.set_settings_fast_compute()

        stream.settings["preprocessing"]["raw_normalization"] = True
        stream.settings["preprocessing"]["preprocessing_order"] = [
            "raw_normalization",
        ]
        stream.settings["postprocessing"]["feature_normalization"] = True
        stream.settings["raw_normalization_settings"][
            "normalization_method"
        ] = {"minmax": True}
        stream.settings["feature_normalization_settings"][
            "normalization_method"
        ] = {
            "mean": False,
            "median": False,
            "zscore": False,
            "zscore-median": False,
            "quantile": False,
            "power": False,
            "robust": False,
            "minmax": True,
        }

        stream.init_stream(
            sfreq=sfreq,
            line_noise=line_noise,
            coord_list=coord_list,
            coord_names=coord_names,
        )

        stream.run(
            data=data,
            out_path_root=PATH_OUT,
            folder_name=RUN_NAME,
        )
