import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from pyneuromodulation import nm_define_nmchannels


class SettingsWrapper:
    def __init__(self, settings_path=None) -> None:
        """initialize settings class with nm_settings.json and setting nm_channels toolbox parameter

        Parameters
        ----------
        settings_path : str, optional
            path to nm_settings.json, by default 'nm_settings.json'
        """
        print("Reading nm_settings.json.")
        self.settings_path = settings_path

        with open(settings_path, encoding="utf-8") as json_file:
            self.settings = json.load(json_file)

        print("Testing settings.")
        self.test_settings()
        self.nm_channels_path = None
        self.nm_channels = None
        self.ind_label = None

    def set_fs_line_noise(self, fs, line_noise) -> None:
        self.settings["fs"] = fs
        self.settings["line_noise"] = line_noise

    def set_channel_names(self, ch_names) -> None:
        self.settings["ch_names"] = ch_names

    def set_channel_types(self, ch_types) -> None:
        self.settings["ch_types"] = ch_types

    def add_coord(self, raw_arr, PATH_GRIDS=None):
        """set coordinate information to settings from RawArray
        The set coordinate positions are set as lists, since np.arrays cannot be saved in json
        Parameters
        ----------
        raw_arr : mne.io.RawArray
        PATH_GRIDS : string, optional
            absolute path to grid_cortex.tsv and grid_subcortex.tsv, by default: None
        """
        if raw_arr.get_montage() is not None:
            self.settings["coord_list"] = np.array(
                list(
                    dict(
                        raw_arr.get_montage().get_positions()["ch_pos"]
                    ).values()
                )
            ).tolist()
            self.settings["coord_names"] = np.array(
                list(
                    dict(
                        raw_arr.get_montage().get_positions()["ch_pos"]
                    ).keys()
                )
            ).tolist()

            self.settings["coord"] = {}
            self.settings["coord"]["cortex_right"] = {}
            self.settings["coord"]["cortex_left"] = {}
            self.settings["coord"]["subcortex_right"] = {}
            self.settings["coord"]["subcortex_left"] = {}
            self.settings["coord"]["cortex_right"]["ch_names"] = [
                self.settings["coord_names"][ch_idx]
                for ch_idx, ch in enumerate(self.settings["coord_list"])
                if (self.settings["coord_list"][ch_idx][0] > 0)
                and ("ECOG" in self.settings["coord_names"][ch_idx])
            ]

            # multiply by 1000 to get m instead of mm
            self.settings["coord"]["cortex_right"]["positions"] = (
                1000
                * np.array(
                    [
                        ch
                        for ch_idx, ch in enumerate(
                            self.settings["coord_list"]
                        )
                        if (self.settings["coord_list"][ch_idx][0] > 0)
                        and ("ECOG" in self.settings["coord_names"][ch_idx])
                    ]
                )
            )

            self.settings["coord"]["cortex_left"]["ch_names"] = [
                self.settings["coord_names"][ch_idx]
                for ch_idx, ch in enumerate(self.settings["coord_list"])
                if (self.settings["coord_list"][ch_idx][0] <= 0)
                and ("ECOG" in self.settings["coord_names"][ch_idx])
            ]
            self.settings["coord"]["cortex_left"]["positions"] = (
                1000
                * np.array(
                    [
                        ch
                        for ch_idx, ch in enumerate(
                            self.settings["coord_list"]
                        )
                        if (self.settings["coord_list"][ch_idx][0] <= 0)
                        and ("ECOG" in self.settings["coord_names"][ch_idx])
                    ]
                )
            )

            self.settings["coord"]["subcortex_right"]["ch_names"] = [
                self.settings["coord_names"][ch_idx]
                for ch_idx, ch in enumerate(self.settings["coord_list"])
                if (self.settings["coord_list"][ch_idx][0] > 0)
                and ("LFP" in self.settings["coord_names"][ch_idx])
            ]
            self.settings["coord"]["subcortex_right"]["positions"] = (
                1000
                * np.array(
                    [
                        ch
                        for ch_idx, ch in enumerate(
                            self.settings["coord_list"]
                        )
                        if (self.settings["coord_list"][ch_idx][0] > 0)
                        and ("LFP" in self.settings["coord_names"][ch_idx])
                    ]
                )
            )

            self.settings["coord"]["subcortex_left"]["ch_names"] = [
                self.settings["coord_names"][ch_idx]
                for ch_idx, ch in enumerate(self.settings["coord_list"])
                if (self.settings["coord_list"][ch_idx][0] <= 0)
                and ("LFP" in self.settings["coord_names"][ch_idx])
            ]
            self.settings["coord"]["subcortex_left"]["positions"] = (
                1000
                * np.array(
                    [
                        ch
                        for ch_idx, ch in enumerate(
                            self.settings["coord_list"]
                        )
                        if (self.settings["coord_list"][ch_idx][0] <= 0)
                        and ("LFP" in self.settings["coord_names"][ch_idx])
                    ]
                )
            )
            if PATH_GRIDS is None:
                self.settings["grid_cortex"] = pd.read_csv(
                    Path(__file__).parent / "grid_cortex.tsv", sep="\t"
                )  # left cortical grid
                self.settings["grid_subcortex"] = pd.read_csv(
                    Path(__file__).parent / "grid_subcortex.tsv", sep="\t"
                )  # left subcortical grid
            else:
                self.settings["grid_cortex"] = pd.read_csv(
                    os.path.join(PATH_GRIDS, "grid_cortex.tsv"), sep="\t"
                )  # left cortical grid
                self.settings["grid_subcortex"] = pd.read_csv(
                    os.path.join(PATH_GRIDS, "grid_subcortex.tsv"), sep="\t"
                )  # left subcortical grid

            if len(self.settings["coord"]["cortex_left"]["positions"]) == 0:
                self.settings["sess_right"] = True
            elif len(self.settings["coord"]["cortex_right"]["positions"]) == 0:
                self.settings["sess_right"] = False
        else:
            self.settings["coord_list"] = None
            self.settings["coord_names"] = None
            self.settings["grid_cortex"] = None
            self.settings["grid_subcortex"] = None

    def set_nm_channels(
        self,
        nm_channels_path=None,
        ch_names=None,
        ch_types=None,
        bads=None,
        ECOG_ONLY=False,
    ) -> None:
        """set nm_channels dataframe; specifies channel names, rereferencing method, ch_type,
        if a channel is used or not, and if a channel should be treated as a label

        Parameters
        ----------
        nm_channels_path : string, optional
            path to nm_channels.tsv file
        ch_names : list, optional
            by default None
        ch_types : list, optional
            BIDS compatible channel types, by default None
        bads : list, optional
            bad channels from raw_arr.info["bads"], by default None
        ECOG_ONLY : boolean, by default False
            if True set 'seeg' or 'dbs' channels unused
        """
        self.nm_channels_path = nm_channels_path
        self.nm_channels = (
            pd.read_csv(self.nm_channels_path, sep="\t")
            if self.nm_channels_path is not None
            and os.path.isfile(self.nm_channels_path)
            else nm_define_nmchannels.set_nm_channels(
                ch_names, ch_types, bads=bads, ECOG_ONLY=ECOG_ONLY
            )
        )
        self.settings["ch_names"] = self.nm_channels[
            self.nm_channels["used"] == 1
        ]["new_name"].tolist()
        self.settings["ch_types"] = self.nm_channels[
            self.nm_channels["used"] == 1
        ]["type"].tolist()

        # feature_idx indexes the nm_channels dataframe which channels are being feature estimated
        self.settings["feature_idx"] = np.where(
            self.nm_channels["used"] & ~self.nm_channels["target"]
        )[0].tolist()
        # If multiple targets exist, select only the first
        self.ind_label = np.where(self.nm_channels["target"] == 1)[0]

    def test_settings(self, verbose=True) -> None:
        """Test if settings are specified correctly in nm_settings.json

        Parameters
        ----------
        verbose: boolean
            set to True if feedback is desired.
        Returns
        -------
        None
        """
        if isinstance(self.settings, dict):
            s = self.settings
        else:
            with open(self.settings_path, encoding="utf-8") as json_file:
                s = json.load(json_file)
            assert isinstance(s, dict)

        # assert (os.path.isdir(s["BIDS_path"]))
        # assert (os.path.isdir(s["out_path"]))
        assert isinstance(s["sampling_rate_features"], (float, int))
        if s["methods"]["project_cortex"] is True:
            assert isinstance(
                s["project_cortex_settings"]["max_dist"], (float, int)
            )
        if s["methods"]["project_subcortex"] is True:
            assert isinstance(
                s["project_subcortex_settings"]["max_dist"], (float, int)
            )
        assert (
            isinstance(value, bool) for value in s["methods"].values()
        ), "Methods must be a boolean value."
        assert any(
            value is True for value in s["methods"].values()
        ), "Set at least one method to True."
        if s["methods"]["raw_resampling"] is True:
            assert isinstance(
                s["raw_resampling_settings"]["resample_freq"], (float, int)
            )
        if s["methods"]["raw_normalization"] is True:
            assert isinstance(
                s["raw_normalization_settings"]["normalization_time"],
                (float, int),
            )
            assert s["raw_normalization_settings"]["normalization_method"] in [
                "mean",
                "median",
                "zscore",
            ]
            assert isinstance(
                s["raw_normalization_settings"]["clip"], (float, int, bool)
            )
        if s["methods"]["feature_normalization"] is True:
            assert isinstance(
                s["feature_normalization_settings"]["normalization_time"],
                (float, int),
            )
            assert s["feature_normalization_settings"][
                "normalization_method"
            ] in ["mean", "median", "zscore", "zscore-median"]
            assert isinstance(
                s["feature_normalization_settings"]["clip"], (float, int, bool)
            )
        if s["methods"]["kalman_filter"] is True:
            assert isinstance(s["kalman_filter_settings"]["Tp"], (float, int))
            assert isinstance(
                s["kalman_filter_settings"]["sigma_w"], (float, int)
            )
            assert isinstance(
                s["kalman_filter_settings"]["sigma_v"], (float, int)
            )
            assert s["kalman_filter_settings"][
                "frequency_bands"
            ], "No frequency bands specified for Kalman filter."
            assert isinstance(
                s["kalman_filter_settings"]["frequency_bands"], list
            ), "Frequency bands for Kalman filter must be specified as a list."
            assert (
                item
                in s["bandpass_filter_settings"]["frequency_ranges"].values()
                for item in s["kalman_filter_settings"]["frequency_bands"]
            ), (
                "Frequency bands for Kalman filter must also be specified in "
                "bandpass_filter_settings."
            )
        if s["methods"]["bandpass_filter"] is True:
            assert isinstance(s["frequency_ranges"], dict)
            assert (
                isinstance(value, list)
                for value in s["frequency_ranges"].values()
            )
            assert (
                len(value) == 2 for value in s["frequency_ranges"].values()
            )
            assert (
                isinstance(value[0], list)
                for value in s["frequency_ranges"].values()
            )
            assert (
                len(value[0]) == 2 for value in s["frequency_ranges"].values()
            )
            assert (
                isinstance(value[1], (float, int))
                for value in s["frequency_ranges"].values()
            )
            assert (
                isinstance(value, bool)
                for value in s["bandpass_filter_settings"][
                    "bandpower_features"
                ].values()
            )
            assert any(
                value is True
                for value in s["bandpass_filter_settings"][
                    "bandpower_features"
                ].values()
            ), "Set at least one bandpower_feature to True."
        if s["methods"]["sharpwave_analysis"] is True:
            assert isinstance(
                s["sharpwave_analysis_settings"]["filter_low_cutoff"],
                (int, float),
            )
            assert isinstance(
                s["sharpwave_analysis_settings"]["filter_high_cutoff"],
                (int, float),
            )
            assert (
                s["sharpwave_analysis_settings"]["filter_high_cutoff"]
                > s["sharpwave_analysis_settings"]["filter_low_cutoff"]
            )
        if s["methods"]["pdc"] is True:
            assert isinstance(s["pdc_settings"]["frequency_ranges"], dict)
            assert (
                key
                in s["bandpass_filter_settings"]["frequency_ranges"].values()
                for key in s["pdc_settings"]["frequency_ranges"].keys()
            ), (
                "Frequency bands for PDC must also be specified in "
                "bandpass_filter_settings."
            )
            assert (
                isinstance(value, list)
                for value in s["pdc_settings"]["frequency_ranges"].values()
            ), "Channels for PDC must be specified as a list."
            assert (
                isinstance(value, list)
                for value in s["pdc_settings"]["frequency_ranges"].values()
            )
            assert isinstance(
                s["pdc_settings"]["model_order"], (str, int)
            ), 'Model order in PDC settings must be either an integer or "auto".'
            if isinstance(s["pdc_settings"]["model_order"], int):
                assert s["pdc_settings"]["model_order"] == "auto", (
                    "Model order in PDC settings must be either an integer "
                    'or "auto".'
                )
                assert isinstance(
                    s["pdc_settings"]["max_order"], int
                ), "Maximum order in PDC settings must be an integer."
            assert isinstance(
                s["pdc_settings"]["num_fft"], (str, int)
            ), 'mum_fft in PDC settings must be either an integer or "auto".'
        if s["methods"]["dtf"] is True:
            assert isinstance(s["dtf_settings"]["frequency_ranges"], dict)
            assert (
                key
                in s["bandpass_filter_settings"]["frequency_ranges"].values()
                for key in s["dtf_settings"]["frequency_ranges"].keys()
            ), (
                "Frequency bands for DTF must also be specified in "
                "bandpass_filter_settings."
            )
            assert (
                isinstance(value, list)
                for value in s["dtf_settings"]["frequency_ranges"].values()
            ), ("Channels for DTF must be " "specified as a list.")
            assert (
                isinstance(value, list)
                for value in s["pdc_settings"]["frequency_ranges"].values()
            )
            assert isinstance(
                s["dtf_settings"]["model_order"], (str, int)
            ), 'Model order in DTF settings must be either an integer or "auto".'
            if isinstance(s["dtf_settings"]["model_order"], int):
                assert s["dtf_settings"]["model_order"] == "auto", (
                    "Model order in DTF settings must be either an integer "
                    'or "auto".'
                )
                assert isinstance(
                    s["dtf_settings"]["max_order"], int
                ), "Maximum order in DTF settings must be an integer."
            assert isinstance(
                s["dtf_settings"]["num_fft"], (str, int)
            ), 'mum_fft in DTF settings must be either an integer or "auto".'
        if verbose:
            print("No Error occurred when testing the settings.")
        return
