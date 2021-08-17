import os
from pathlib import Path
import mne

from pyneuromodulation import nm_IO, nm_projection, nm_generator, nm_rereference, \
    nm_run_analysis, nm_features, nm_resample, nm_settings


class NM_BIDS:

    def __init__(self, PATH_RUN, PATH_NM_CHANNELS=None, PATH_SETTINGS=None,
                 PATH_ANNOTATIONS=None, LIMIT_DATA=False, LIMIT_LOW=0, LIMIT_HIGH=25000,
                 ECOG_ONLY=False, verbose=True) -> None:
        """Start feature estimation by reading settings, creating or reading
        nm_channels.csv file with default rereference function (ECoG CAR; depth LFP bipolar)
        Parameters

        Parameters
        ----------
        PATH_RUN : string
            absolute path to run file
        PATH_NM_CHANNELS : string, optional
            absolute path to nm_channels.csv file, by default None
        PATH_SETTINGS : string, optional
            absolute path to settings.json file, by default None
        PATH_ANNOTATIONS : string, optional
            absolute path to folder with mne annotations.txt, by default None
        LIMIT_DATA : bool, optional
            restrict processsing samples, by default False
        LIMIT_LOW : int, optional
            lower sample limit, by default 0
        LIMIT_HIGH : int, optional
            upper sample limit, by default 120000
        ECOG_ONLY : bool, optional
            if True, select ECoG channel only, by default False
        verbose : bool, optional
            print real time simulation progress, by default True
        """

        self.PATH_SETTINGS = PATH_SETTINGS
        self.PATH_NM_CHANNELS = PATH_NM_CHANNELS
        self.PATH_RUN = PATH_RUN
        self.PATH_ANNOTATIONS = PATH_ANNOTATIONS
        self.ECOG_ONLY = ECOG_ONLY
        self.verbose = verbose
        self.settings_wrapper = None
        self.raw_arr = None
        self.annot = None
        self.fs = None
        self.line_noise = None
        self.projection = None
        self.generator = None
        self.resample = None
        self.features = None
        self.run_analysis = None

        self.set_settings(self.PATH_SETTINGS)

        # read BIDS data
        self.raw_arr, self.raw_arr_data, self.fs, self.line_noise = nm_IO.read_BIDS_data(
            self.PATH_RUN, self.settings_wrapper.settings['BIDS_path'])

        if self.PATH_ANNOTATIONS is not None:
            self.set_annotations(self.PATH_ANNOTATIONS)

        self.set_nm_channels()
        self.set_projection()

        if LIMIT_DATA is True:
            self.limit_data(LIMIT_LOW, LIMIT_HIGH)

        self.set_generator()
        self.set_rereferencing()
        self.set_resampling()
        self.set_features()
        self.set_run()

    def run_bids(self):
        self.call_run()
        self.add_labels()
        self.save_features()

    def set_settings(self, PATH_SETTINGS=None):
        if PATH_SETTINGS is None and self.PATH_SETTINGS is not None:
            self.PATH_SETTINGS = PATH_SETTINGS
        elif PATH_SETTINGS is None:
            PATH_SETTINGS = os.path.join(os.path.dirname(nm_IO.__file__), 'nm_settings.json')
            self.settings_wrapper = nm_settings.SettingsWrapper(PATH_SETTINGS)
        else:
            # PATH_SETTINGS is parametrized
            self.settings_wrapper = nm_settings.SettingsWrapper(settings_path=self.PATH_SETTINGS)

    def set_annotations(self):
        try:
            self.annot = mne.read_annotations(os.path.join(self.PATH_ANNOTATIONS,
                                              os.path.basename(self.PATH_RUN)[:-5]+".txt"))
            self.raw_arr.set_annotations(self.annot)
            # annotations starting with "BAD" are omitted with reject_by_annotations 'omit' param
            self.raw_arr_data = self.raw_arr.get_data(reject_by_annotation='omit')
        except FileNotFoundError:
            print("Annotations file could not be found")
            print("expected location: "+str(os.path.join(self.PATH_ANNOTATIONS,
                                            os.path.basename(self.PATH_RUN)[:-5]+".txt")))
            pass

    def set_projection(self):
        # (if available) add coordinates to settings
        if self.raw_arr.get_montage() is not None:
            self.settings_wrapper.add_coord(self.raw_arr.copy())

        if any((self.settings_wrapper.settings["methods"]["project_cortex"],
                self.settings_wrapper.settings["methods"]["project_subcortex"])):
            self.projection = nm_projection.Projection(self.settings_wrapper.settings)
        else:
            self.projection = None

    def set_nm_channels(self, PATH_NM_CHANNELS=None):
        if PATH_NM_CHANNELS is not None:
            self.PATH_NM_CHANNELS = PATH_NM_CHANNELS
        # read nm_channels.csv or create nm_channels if None specified
        self.settings_wrapper.set_nm_channels(nm_channels_path=self.PATH_NM_CHANNELS, ch_names=self.raw_arr.ch_names,
                                              ch_types=self.raw_arr.get_channel_types(), bads=self.raw_arr.info["bads"],
                                              ECOG_ONLY=self.ECOG_ONLY)
        self.settings_wrapper.set_fs_line_noise(self.fs, self.line_noise)

    def limit_data(self, LIMIT_LOW=0, LIMIT_HIGH=120000):
        """Reduce timing for faster test completion

        Parameters
        ----------
        LIMIT_LOW : int, optional
            Start sample limit, by default 0
        LIMIT_HIGH : int, optional
            End sample limit, by default 120000
        """

        self.LIMIT_LOW = LIMIT_LOW
        self.LIMIT_HIGH = LIMIT_HIGH
        self.raw_arr_data = self.raw_arr_data[:, LIMIT_LOW:LIMIT_HIGH]

    def set_generator(self):
        """initialize generator for run function
        """
        self.gen = nm_generator.ieeg_raw_generator(self.raw_arr_data, self.settings_wrapper.settings)

    def set_rereferencing(self):
        """initialize rereferencing
        """
        if self.settings_wrapper.settings["methods"]["re_referencing"] is True:
            self.rereference = nm_rereference.RT_rereference(
                self.settings_wrapper.nm_channels, split_data=False)
        else:
            self.rereference = None
            # reset nm_channels from default values
            self.settings_wrapper.nm_channels["rereference"] = None
            self.settings_wrapper.nm_channels["new_name"] = self.settings_wrapper.nm_channels["name"]

    def set_resampling(self):
        """define resampler for faster feature estimation
        """
        if self.settings_wrapper.settings["methods"]["raw_resampling"] is True:
            self.resample = nm_resample.Resample(self.settings_wrapper.settings)
        else:
            self.resample = None

    def set_features(self):
        """initialize feature class from settings
        """
        self.features = nm_features.Features(self.settings_wrapper.settings, verbose=self.verbose)

    def set_run(self):
        """initialize run object
        """
        self.run_analysis = nm_run_analysis.Run(
            self.features, self.settings_wrapper.settings, self.rereference, self.projection,
            self.resample, verbose=self.verbose)

    def call_run(self):
        while True:
            ieeg_batch = next(self.gen, None)
            if ieeg_batch is not None:
                self.run_analysis.run(ieeg_batch)
            else:
                break

    def add_labels(self):
        """add resampled labels to feature dataframe
        """
        self.df_features = nm_IO.add_labels(
            self.run_analysis.feature_arr, self.settings_wrapper, self.raw_arr_data)

    def save_features(self):
        """save settings.json, nm_channels.csv and features.csv
           and pickled run_analysis including projections
        """
        self.run_analysis.feature_arr = self.df_features  # here the potential label stream is added
        nm_IO.save_features_and_settings(df_=self.df_features, run_analysis=self.run_analysis,
                                         folder_name=os.path.basename(self.PATH_RUN)[:-5],
                                         settings_wrapper=self.settings_wrapper)
