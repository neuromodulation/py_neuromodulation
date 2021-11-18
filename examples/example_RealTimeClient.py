from os import path
import pathlib
from pyneuromodulation import nm_IO, nm_projection, nm_generator, nm_rereference, \
    nm_run_analysis, nm_features, nm_resample, nm_settings


class RealTimePyNeuro():

    def __init__(self) -> None:

        # set settings path to current directory, just nm_settings.json
        self.PATH_SETTINGS = path.join(pathlib.Path(__file__).parent.resolve(),\
            "rt_example", "nm_settings.json")

        # set nmchannels.tsv path to curent file directory
        self.PATH_NM_CHANNELS = path.join(pathlib.Path(__file__).parent.resolve(),\
            "rt_example", "nm_channels.csv")

        # set it here
        self.fs = 250
        self.line_noise = 50
        
        self.resample = None
        self.features = None
        self.run_analysis = None
        self.rereference = None
        self.projection = None
        self.verbose = True
        # Settings
        self.settings_wrapper = nm_settings.SettingsWrapper(self.PATH_SETTINGS)

        self.settings_wrapper.set_nm_channels(nm_channels_path=self.PATH_NM_CHANNELS)
        self.settings_wrapper.set_fs_line_noise(self.fs, self.line_noise)

        # leave out projection out for now

        # Rereferencing
        if self.settings_wrapper.settings["methods"]["re_referencing"] is True:
            self.rereference = nm_rereference.RT_rereference(
                self.settings_wrapper.nm_channels, split_data=False)
        else:
            self.rereference = None
            # reset nm_channels from default values
            self.settings_wrapper.nm_channels["rereference"] = None
            self.settings_wrapper.nm_channels["new_name"] = self.settings_wrapper.nm_channels["name"]

        # Resampling
        if self.settings_wrapper.settings["methods"]["raw_resampling"] is True:
            self.resample = nm_resample.Resample(self.settings_wrapper.settings)
        else:
            self.resample = None

        # init features
        self.features = nm_features.Features(self.settings_wrapper.settings, verbose=self.verbose)

        self.run_analysis = nm_run_analysis.Run(
            self.features, self.settings_wrapper.settings, self.rereference, self.projection,
            self.resample, verbose=self.verbose)

    def call_run(self, ieeg_batch):
        self.run_analysis.run(ieeg_batch)

        # return last estimated data batch by run_analysis
        return self.run_analysis.feature_arr.iloc[-1,:]

