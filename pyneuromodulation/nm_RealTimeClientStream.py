from os import path
import pathlib
import numpy as np
import pandas as pd

from pyneuromodulation import \
    (nm_projection,
    nm_rereference,
    nm_run_analysis,
    nm_features,
    nm_resample,
    nm_settings,
    nm_stream)


class RealTimePyNeuro(nm_stream.PNStream):

    resample: nm_resample.Resample = None
    features: nm_features.Features = None
    run_analysis: nm_run_analysis.Run = None
    rereference: nm_rereference.RT_rereference = None
    projection: nm_projection.Projection = None
    settings: nm_settings.SettingsWrapper = None
    fs: float = None
    line_noise: float = None
    verbose: bool = False

    def __init__(self,
            PATH_SETTINGS: str = path.join(pathlib.Path(__file__).parent.resolve(),\
                                          "rt_example", "nm_settings.json"),
            PATH_NM_CHANNELS: str = path.join(pathlib.Path(__file__).parent.resolve(),\
                                              "rt_example", "nm_channels.csv"),
            fs: float = 128,
            line_noise: float = 50) -> None:

        self.PATH_SETTINGS = PATH_SETTINGS

        self.PATH_NM_CHANNELS = PATH_NM_CHANNELS

        self.set_fs(fs)
        self.set_linenoise(line_noise)

        self.set_settings(self.PATH_SETTINGS)

        self.set_channels(self.PATH_NM_CHANNELS)

        self.settings_wrapper.set_fs_line_noise(self.fs, self.line_noise)

        self.set_rereference()

        self.set_resampling()

        self.set_features()

        self.set_run()

    def run(self, ieeg_batch: np.array) -> pd.Series:
        self.run_analysis.run(ieeg_batch)

        # return last estimated data batch by run_analysis
        return self.run_analysis.feature_arr.iloc[-1, :]

    def set_rereference(self) -> None:
        if self.settings_wrapper.settings["methods"]["re_referencing"] is True:
            self.rereference = nm_rereference.RT_rereference(
                self.settings_wrapper.nm_channels, split_data=False)
        else:
            self.rereference = None
            # reset nm_channels from default values
            self.settings_wrapper.nm_channels["rereference"] = None
            self.settings_wrapper.nm_channels["new_name"] = self.settings_wrapper.nm_channels["name"]

    def set_resampling(self) -> None:
        if self.settings_wrapper.settings["methods"]["raw_resampling"] is True:
            self.resample = nm_resample.Resample(self.settings_wrapper.settings)
        else:
            self.resample = None

    def set_settings(self, PATH_SETTINGS: str) -> None:
        self.settings_wrapper = nm_settings.SettingsWrapper(PATH_SETTINGS)

    def set_fs(self, fs: int) -> None:
        self.fs = fs

    def set_linenoise(self, line_noise: int) -> None:
        self.line_noise = line_noise

    def set_channels(self, PATH_CHANNELS: str) -> None:
        self.settings_wrapper.set_nm_channels(nm_channels_path=PATH_CHANNELS)

    def set_features(self) -> None:
        self.features = nm_features.Features(self.settings_wrapper.settings, verbose=self.verbose)

    def set_run(self) -> None:
        self.run_analysis = nm_run_analysis.Run(
            self.features, self.settings_wrapper.settings, self.rereference, self.projection,
            self.resample, verbose=self.verbose)
