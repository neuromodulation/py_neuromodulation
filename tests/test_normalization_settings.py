import unittest
import os
import py_neuromodulation as nm
from py_neuromodulation import (
    nm_define_nmchannels,
    nm_IO,
)


class TestNormSettings(unittest.TestCase):

    def set_up(self):
        """
        Load BIDS data, create stream.
        Returns necessary variables for testing different settings.
        :return: stream, sfreq, line_noise, coord_list, coord_names, data, PATH_OUT, RUN_NAME
        """
        SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath('.')), "examples")

        sub = "testsub"
        ses = "EphysMedOff"
        task = "buttonpress"
        run = 0
        datatype = "ieeg"

        # Define run name and access paths in the BIDS format.
        RUN_NAME = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}"

        PATH_RUN = os.path.join(
            (os.path.join(SCRIPT_DIR, "data")),
            f"sub-{sub}",
            f"ses-{ses}",
            datatype,
            RUN_NAME,
        )
        PATH_BIDS = os.path.join(SCRIPT_DIR, "data")

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

        # Provide a path for the output data. Each re-referencing method has their PATH_OUT
        PATH_OUT = os.path.join(SCRIPT_DIR, "data", "derivatives", "test_normalization")
        nm_channels = nm_define_nmchannels.set_channels(
            ch_names=raw.ch_names,
            ch_types=raw.get_channel_types(),
            reference='default',
            bads=raw.info["bads"],
            new_names="default",
            used_types=("ecog",),  # We focus only on LFP data
            target_keywords=("SQUARED_ROTATION",),
        )

        stream = nm.Stream(
            settings=None,
            nm_channels=nm_channels,
            path_grids=None,
            verbose=False,
        )
        return stream, sfreq, line_noise, coord_list, coord_names, data, nm_channels, PATH_OUT, RUN_NAME

    def test_fast_compute_settings(self):
        """
        Try if normalization on fast compute settings works.
        No assertion in the end, only want to see if it raises any errors
        """
        stream, sfreq, line_noise, coord_list, coord_names, data, nm_channels, PATH_OUT, RUN_NAME = self.set_up()

        stream.set_settings_fast_compute()
        stream.init_stream(
            sfreq=sfreq,
            line_noise=line_noise,
            coord_list=coord_list,
            coord_names=coord_names
        )

        stream.run(
            data=data,
            out_path_root=PATH_OUT,
            folder_name=RUN_NAME,
        )

    def test_quantile_norm(self):
        stream, sfreq, line_noise, coord_list, coord_names, data, nm_channels, PATH_OUT, RUN_NAME = self.set_up()

        stream.set_settings_fast_compute()

        stream.settings['preprocessing']['raw_normalization'] = True
        stream.settings['preprocessing']['preprocessing_order'] = ["raw_normalization",]
        stream.settings['postprocessing']['feature_normalization'] = True
        stream.settings[ "raw_normalization_settings"]["normalization_method"] = {
            "mean": False,
            "median": False,
            "zscore": False,
            "zscore-median": False,
            "quantile": True,
            "power": False,
            "robust": False,
            "minmax": False
                }
        stream.settings[ "feature_normalization_settings"]["normalization_method"] = {
            "mean": False,
            "median": False,
            "zscore": False,
            "zscore-median": False,
            "quantile": True,
            "power": False,
            "robust": False,
            "minmax": False
                }

        stream.init_stream(
            sfreq=sfreq,
            line_noise=line_noise,
            coord_list=coord_list,
            coord_names=coord_names
        )

        stream.run(
            data=data,
            out_path_root=PATH_OUT,
            folder_name=RUN_NAME,
        )


    def test_zscore_median_norm(self):
        stream, sfreq, line_noise, coord_list, coord_names, data, nm_channels, PATH_OUT, RUN_NAME = self.set_up()

        stream.set_settings_fast_compute()

        stream.settings['preprocessing']['raw_normalization'] = True
        stream.settings['preprocessing']['preprocessing_order'] = ["raw_normalization",]
        stream.settings['postprocessing']['feature_normalization'] = True
        stream.settings[ "raw_normalization_settings"]["normalization_method"] = {
            "mean": False,
            "median": False,
            "zscore": False,
            "zscore-median": True,
            "quantile": False,
            "power": False,
            "robust": False,
            "minmax": False
                }
        stream.settings[ "feature_normalization_settings"]["normalization_method"] = {
            "zscore-median": True,
                }

        stream.init_stream(
            sfreq=sfreq,
            line_noise=line_noise,
            coord_list=coord_list,
            coord_names=coord_names
        )

        stream.run(
            data=data,
            out_path_root=PATH_OUT,
            folder_name=RUN_NAME,
        )


    def test_minmax_norm(self):
        stream, sfreq, line_noise, coord_list, coord_names, data, nm_channels, PATH_OUT, RUN_NAME = self.set_up()

        stream.set_settings_fast_compute()

        stream.settings['preprocessing']['raw_normalization'] = True
        stream.settings['preprocessing']['preprocessing_order'] = ["raw_normalization",]
        stream.settings['postprocessing']['feature_normalization'] = True
        stream.settings[ "raw_normalization_settings"]["normalization_method"] = {
            "minmax": True
                }
        stream.settings[ "feature_normalization_settings"]["normalization_method"] = {
            "mean": False,
            "median": False,
            "zscore": False,
            "zscore-median": False,
            "quantile": False,
            "power": False,
            "robust": False,
            "minmax": True
                }

        stream.init_stream(
            sfreq=sfreq,
            line_noise=line_noise,
            coord_list=coord_list,
            coord_names=coord_names
        )

        stream.run(
            data=data,
            out_path_root=PATH_OUT,
            folder_name=RUN_NAME,
        )
