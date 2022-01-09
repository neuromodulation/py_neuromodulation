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
    nm_eval_timing,
)


class Run:
    def __init__(
        self,
        features: nm_features.Features,
        settings: dict,
        reference: nm_rereference.RT_rereference,
        projection: nm_projection.Projection,
        resample: nm_resample.Resample,
        notch_filter : nm_notch_filter.NotchFilter,
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
        notch_filter : nm_notch_filter.NotchFilter,
            Notch Filter object, needs to be instantiated beforehand
        verbose : boolean
            if True, print out signal processed and computation time
        """

        self.features = features
        self.feature_names = None
        self.features_previous = None
        self.features_current = None
        self.raw_arr = None
        self.reference = reference
        self.resample = resample
        self.notch_filter = notch_filter
        self.projection = projection 
        self.project: bool = False if projection is None else \
            any((self.projection.project_cortex, self.projection.project_subcortex))
        self.settings = settings
        self.fs_new = int(settings["sampling_rate_features"])
        self.fs = features.fs
        self.line_noise = features.line_noise
        self.feature_idx = feature_idx
        self.sample_add = int(self.fs / self.fs_new)
        self.verbose = verbose
        self.offset = max(
            [
                value
                for value in settings["bandpass_filter_settings"][
                    "segment_lengths"
                ].values()
            ]
        )  # ms

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
            #ieeg_batch = nm_notch_filter.notch_filter(
            #    ieeg_batch, self.fs, self.line_noise
            #)
            ieeg_batch = self.notch_filter.filter_data(ieeg_batch)

        # normalize raw data
        if self.settings["methods"]["raw_normalization"] is True:
            ieeg_batch, self.raw_arr = nm_normalization.normalize_raw(
                current=ieeg_batch,
                previous=self.raw_arr,
                normalize_samples=self.raw_normalize_samples,
                sample_add=self.sample_add,
                method=self.settings["raw_normalization_settings"][
                    "normalization_method"
                ],
                clip=self.settings["raw_normalization_settings"]["clip"],
            )

        # calculate features
        features_dict = self.features.estimate_features(ieeg_batch)
        features_values = np.array(list(features_dict.values()), dtype=float)

        # normalize features
        if self.settings["methods"]["feature_normalization"]:
            (
                features_values,
                self.features_previous,
            ) = nm_normalization.normalize_features(
                current=features_values,
                previous=self.features_previous,
                normalize_samples=self.feat_normalize_samples,
                method=self.settings["feature_normalization_settings"][
                    "normalization_method"
                ],
                clip=self.settings["feature_normalization_settings"]["clip"],
            )

        self.features_current = pd.Series(
            data=features_values, index=features_dict.keys(), dtype=float
        )

        # project features to grid
        if self.project:
            self.features_current = self.projection.project_features(
                self.features_current
            )

        # add sample counts
        if self.cnt_samples == 0:
            self.cnt_samples += int(self.fs)
        else:
            self.cnt_samples += self.sample_add

        if self.verbose is True:
            print(
                "Last batch took: "
                + str(np.round(time() - start_time, 2))
                + " seconds"
            )

        # if self.cnt_samples > 4000:
        #    nm_eval_timing.NM_Timer(self)

        return self.features_current
