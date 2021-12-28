from time import time
import numpy as np
import pandas as pd

from py_neuromodulation import (
    nm_features,
    nm_notch_filter,
    nm_normalization,
    nm_projection,
    nm_rereference,
    nm_resample,
    nm_eval_timing
)


class Run:

    def __init__(self, features: nm_features.Features,
        settings: dict, reference:
        nm_rereference.RT_rereference,
    def __init__(
        self,
        features: nm_features.Features,
        settings: dict,
        reference: nm_rereference.RT_rereference,
        projection: nm_projection.Projection,
        resample: nm_resample.Resample,
        nm_channels: pd.DataFrame,
        coords: dict,
        sess_right: bool,
        verbose: bool,
        feature_idx: list,
    ) -> None:
        """Initialize run class

        Parameters
        ----------
        features : features.py object
            Feature_df object (needs to be initialized beforehand)
        settings : dict
            dictionary of settings such as "seglengths" or "frequencyranges"
        reference : reference.py object
            Rereference object (needs to be initialized beforehand), by default None
        projection : projection.py object
            projection object (needs to be initialized beforehand), by default None
        resample : resample.py object
            Resample object (needs to be initialized beforehand), by default None
        verbose : boolean
            if True, print out signal processed and computation time
        """

        self.features = features
        self.feature_names = None
        self.features_previous = None
        self.features_current = None
        self.raw_arr = None
        self.proj_cortex_bool:bool = settings["methods"]["project_cortex"]
        self.proj_subcortex_bool:bool = settings["methods"]["project_subcortex"]
        self.dat_cortex: np.array = None
        self.dat_subcortex: np.array = None
        self.reference = reference
        self.resample = resample
        self.nm_channels = nm_channels
        self.projection = projection
        self.coords = coords
        self.sess_right = sess_right
        self.settings = settings
        self.fs_new = int(settings["sampling_rate_features"])
        self.fs = features.fs
        self.line_noise = features.line_noise
        self.feature_idx = feature_idx
        self.sample_add = int(self.fs / self.fs_new)
        print(self.sample_add)
        self.verbose = verbose
        self.offset = max(
            [
                value
                for value in settings["bandpass_filter_settings"]["segment_lengths"].values()
            ]
        )  # ms

        if settings["methods"]["project_cortex"] is True:
            self.idx_chs_ecog = []  # feature series indexes for dbs/lfp channels
            self.names_chs_ecog = []  # feature series name of ecog features
            self.ecog_channels = [
                self.nm_channels.new_name[ch_idx]
                for ch_idx, ch in enumerate(self.nm_channels.type) if ch == "ecog"
            ]
        if settings["methods"]["project_subcortex"] is True:
            self.idx_chs_lfp = []  # feature series indexes for ecog channels
            self.names_chs_lfp = []  # feature series name of lfp features
            #  mind here that coord["subcortex_left/right"] is based on the "LFP" substring in the channel
            self.lfp_channels = self.coords["subcortex_right"]["ch_names"] \
                if self.sess_right is True \
                else self.coords["subcortex_left"]["ch_names"]

        if settings["methods"]["raw_normalization"] is True:
            self.raw_normalize_samples = int(
                settings["raw_normalization_settings"]["normalization_time"]
                * self.fs
            )

        if settings["methods"]["feature_normalization"] is True:
            self.feat_normalize_samples = int(
                settings["feature_normalization_settings"][
                    "normalization_time"
                ]
                * self.fs_new
            )

        self.cnt_samples = 0

    def process_data(self, ieeg_batch):
        """Given a new data batch, estimate features and return them to the streamer

        Parameters
        ----------
        ieeg_batch : np.ndarray
            Current batch of raw data

        Returns
        -------
        None
        """
        start_time = time()

        # re-reference
        if self.settings["methods"]["re_referencing"] is True:
            ieeg_batch = self.reference.rereference(ieeg_batch)

        ieeg_batch = ieeg_batch[self.feature_idx, :]

        # resample
        if self.settings["methods"]["raw_resampling"] is True:
            ieeg_batch = self.resample.raw_resampling(ieeg_batch)

        # notch filter
        if self.settings["methods"]["notch_filter"] is True:
            ieeg_batch = nm_notch_filter.notch_filter(
                ieeg_batch,
                self.fs,
                self.line_noise
            )

        # normalize raw data
        if self.settings["methods"]["raw_normalization"] is True:
            if self.cnt_samples == 0:
                self.raw_arr = ieeg_batch
            else:
                self.raw_arr = np.concatenate(
                    (self.raw_arr, ieeg_batch[:, -self.sample_add:]), axis=1)
            ieeg_batch = nm_normalization.normalize_raw(
                self.raw_arr, self.raw_normalize_samples, self.fs,
                self.settings["raw_normalization_settings"]["normalization_method"],
                self.settings["raw_normalization_settings"]["clip"])

        self.features_current = pd.Series(
            self.features.estimate_features(ieeg_batch), dtype=float
        )

        if self.settings["methods"]["feature_normalization"]:
            (
                self.features_current.iloc[:],
                self.features_previous,
            ) = nm_normalization.normalize_features(
                current=self.features_current.to_numpy(),
                previous=self.features_previous,
                normalize_samples=self.feat_normalize_samples,
                method=self.settings["feature_normalization_settings"][
                    "normalization_method"
                ],
                clip=self.settings["feature_normalization_settings"]["clip"],
            )

        if self.cnt_samples == 0:
            self.cnt_samples += int(self.fs)

            if any((self.proj_cortex_bool, self.proj_subcortex_bool)):
                self.features_current = self.init_projection_run(self.features_current)

        else:
            self.cnt_samples += self.sample_add

            if any((self.proj_cortex_bool, self.proj_subcortex_bool)):
                self.features_current = self.next_projection_run(self.features_current)

        if self.verbose is True:
            print(
                "Last batch took: "
                + str(np.round(time() - start_time, 2))
                + " seconds"
            )

        if self.cnt_samples > 4000:
            nm_eval_timing.NM_Timer(self)

        return self.features_current

    def init_projection_run(self, feature_series):
        """Initialize indexes for respective channels 
        in feature series computed by nm_features.py
        """

        #  here it is assumed that only one hemisphere is recorded at a time!
        if self.proj_cortex_bool:
            for ecog_channel in self.ecog_channels:
                self.idx_chs_ecog.append(
                    [
                        ch_idx
                        for ch_idx, ch in enumerate(feature_series.keys())
                        if ch.startswith(ecog_channel)
                    ]
                )
                self.names_chs_ecog.append(
                    [
                        ch
                        for _, ch in enumerate(feature_series.keys())
                        if ch.startswith(ecog_channel)
                    ]
                )

        if self.proj_subcortex_bool:
            # for lfp_channels select here only the ones from the correct hemisphere!
            for lfp_channel in self.lfp_channels:
                self.idx_chs_lfp.append(
                    [
                        ch_idx
                        for ch_idx, ch in enumerate(feature_series.keys())
                            if ch.startswith(lfp_channel)
                    ]
                )
                self.names_chs_lfp.append(
                    [
                        ch
                        for _, ch in enumerate(feature_series.keys())
                            if ch.startswith(lfp_channel)
                    ]
                )
        
        # get feature_names; given by ECoG sequency of features
        self.feature_names = [feature_name[len(self.ecog_channels[0])+1:] \
                              for feature_name in self.names_chs_ecog[0]]

        return self.next_projection_run(feature_series)

    def next_projection_run(self, feature_series: pd.Series):
        """Project data, given idx_chs_ecog/stn"""

        if self.proj_cortex_bool:
            self.dat_cortex = np.vstack(
                [feature_series.iloc[idx_ch].values for idx_ch in
                 self.idx_chs_ecog])
        if self.proj_subcortex_bool:
            self.dat_subcortex = np.vstack(
                [feature_series.iloc[idx_ch].values for idx_ch in
                 self.idx_chs_lfp])

        # project data
        proj_cortex_array, proj_subcortex_array = \
            self.projection.get_projected_cortex_subcortex_data(
                self.dat_cortex,
                self.dat_subcortex
            )

        # proj_cortex_array has shape grid_points x feature_number
        if self.proj_cortex_bool:
            feature_series = feature_series.append(
                pd.Series(
                    {
                        "gridcortex_" + str(act_grid_point) + "_" + feature_name :
                        proj_cortex_array[act_grid_point, feature_idx] 
                            for feature_idx, feature_name in enumerate(self.feature_names) 
                            for act_grid_point in self.projection.active_cortex_gridpoints
                    }
                )
            )
        if self.proj_subcortex_bool:
            feature_series = feature_series.append(
                pd.Series(
                    {
                        "gridsubcortex_" + str(act_grid_point) + "_" + feature_name :
                        proj_subcortex_array[act_grid_point, feature_idx] 
                            for feature_idx, feature_name in enumerate(self.feature_names) 
                            for act_grid_point in self.projection.active_subcortex_gridpoints
                    }
                )
            )

        return feature_series
