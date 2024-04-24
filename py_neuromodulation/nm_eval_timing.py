import timeit
from . import nm_oscillatory
import numpy as np
from py_neuromodulation.nm_normalization import normalize_features, normalize_raw
from py_neuromodulation.nm_filter import apply_filter
from py_neuromodulation.nm_oscillatory import STFT, BandPower

#from py_neuromodulation import (
    # The following 2 files do not exist in the project, maybe there were deleted?
    # nm_stft, 
    # nm_bandpower,
#)


class NM_Timer:
    def __init__(self, analyzer) -> None:
        self.analyzer = analyzer

        self.get_timings()

    def get_timings(self, number_repeat=1000):

        features_ = {}
        ch_idx = 0
        fs = self.analyzer.fs
        ch_name = "ECOG_L_1_SMC_AT"
        N_CH_BEFORE_REREF = 15  # 2
        N_CH_AFTER_REREF = 11  # 2
        data = np.random.random([N_CH_BEFORE_REREF, fs])

        dict_timings = {}

        if self.analyzer.settings["methods"]["notch_filter"]:
            dict_timings["time_rereference"] = (
                timeit.timeit(
                    lambda: self.analyzer.reference.rereference(data),
                    number=number_repeat,
                )
                / number_repeat
            )

        data = np.random.random([N_CH_AFTER_REREF, fs])

        if self.analyzer.settings["methods"]["raw_resampling"]:
            dict_timings["time_resample"] = (
                timeit.timeit(
                    lambda: self.analyzer.resample.raw_resampling(data),
                    number=number_repeat,
                )
                / number_repeat
            )

            data = np.random.random(
                [
                    N_CH_AFTER_REREF,
                    self.analyzer.settings["raw_resampling_settings"][
                        "resample_freq"
                    ],
                ]
            )

        if self.analyzer.settings["methods"]["notch_filter"]:
            dict_timings["time_notchfilter"] = (
                timeit.timeit(
                    lambda: self.analyzer.notch_filter.filter_data(data),
                    number=number_repeat,
                )
                / number_repeat
            )

        if self.analyzer.settings["methods"]["raw_normalization"]:
            dict_timings["time_norm_raw"] = (
                timeit.timeit(
                    lambda: normalize_raw(
                        current=data,
                        previous=data.T,
                        normalize_samples=int(
                            self.analyzer.settings[
                                "raw_normalization_settings"
                            ]["normalization_time"]
                            * self.analyzer.fs
                        ),
                        sample_add=int(self.analyzer.fs / self.analyzer.fs_new),
                        method=self.analyzer.settings[
                            "raw_normalization_settings"
                        ]["normalization_method"],
                        clip=self.analyzer.settings[
                            "raw_normalization_settings"
                        ]["clip"],
                    ),
                    number=number_repeat,
                )
                / number_repeat
            )

        features_previous = self.analyzer.features_previous
        features_current = self.analyzer.features_current.iloc[
            : features_previous.shape[1]
        ]

        if self.analyzer.settings["methods"]["feature_normalization"]:
            dict_timings["time_feature_norm"] = (
                timeit.timeit(
                    lambda: normalize_features(
                        current=features_current.to_numpy(),
                        previous=features_previous,
                        normalize_samples=self.analyzer.feat_normalize_samples,
                        method=self.analyzer.settings[
                            "feature_normalization_settings"
                        ]["normalization_method"],
                        clip=self.analyzer.settings[
                            "feature_normalization_settings"
                        ]["clip"],
                    ),
                    number=number_repeat,
                )
                / number_repeat
            )

        if self.analyzer.settings["methods"]["project_cortex"]:
            dict_timings["time_projection"] = (
                timeit.timeit(
                    lambda: self.analyzer.projection.project_features(
                        features_current
                    ),
                    number=number_repeat,
                )
                / number_repeat
            )

        if self.analyzer.settings["methods"]["bandpass_filter"]:
            dict_timings["time_applyfilterband"] = (
                timeit.timeit(
                    lambda: self.analyzer.features.bandpass_filter.filter_data(
                        data,
                    ),
                    number=number_repeat,
                )
                / number_repeat
            )

        if self.analyzer.settings["methods"]["sharpwave_analysis"]:
            dict_timings["time_sw"] = (
                timeit.timeit(
                    lambda: self.analyzer.features.sw_features.get_sharpwave_features(
                        features_, data[ch_idx, -100:], ch_name
                    ),
                    number=number_repeat,
                )
                / number_repeat
            )

        if self.analyzer.settings["methods"]["stft"]:
            dict_timings["time_stft"] = (
                timeit.timeit(
                    lambda: nm_stft.get_stft_features(
                        features_,
                        self.analyzer.features.s,
                        self.analyzer.features.fs,
                        data[ch_idx, :],
                        self.analyzer.features.KF_dict,
                        ch_name + "-avgref",
                        self.analyzer.features.f_ranges,
                        self.analyzer.features.fband_names,
                    ),
                    number=number_repeat,
                )
                / number_repeat
            )

        if self.analyzer.settings["methods"]["fft"]:
            dict_timings["time_fft"] = (
                timeit.timeit(
                    lambda: nm_oscillatory.get_fft_features(
                        features_,
                        self.analyzer.features.s,
                        self.analyzer.features.fs,
                        data[ch_idx, :],
                        self.analyzer.features.KF_dict,
                        ch_name,
                        self.analyzer.features.f_ranges,
                        self.analyzer.features.fband_names,
                    ),
                    number=number_repeat,
                )
                / number_repeat
            )

        if self.analyzer.settings["methods"]["bandpass_filter"]:
            seglengths = np.floor(
                self.analyzer.fs
                / 1000
                * np.array(
                    [
                        value
                        for value in self.analyzer.features.s[
                            "bandpass_filter_settings"
                        ]["segment_lengths"].values()
                    ]
                )
            ).astype(int)

            dat_filtered = apply_filter(
                data, self.analyzer.features.filter_fun
            )  # shape (bands, time)
            dict_timings["time_bandpass_filter"] = (
                timeit.timeit(
                    lambda: nm_bandpower.get_bandpower_features(
                        features_,
                        self.analyzer.features.s,
                        seglengths,
                        dat_filtered,
                        self.analyzer.features.KF_dict,
                        ch_name,
                        ch_idx,
                    ),
                    number=number_repeat,
                )
                / number_repeat
            )

        if self.analyzer.settings["methods"]["coherence"]:
            coh_obj = self.analyzer.features.coherence_objects[0]
            dict_timings["time_coherence"] = (
                timeit.timeit(
                    lambda: coh_obj.get_coh(
                        features_,
                        data[coh_obj.ch_1_idx, :],
                        data[coh_obj.ch_2_idx, :],
                    ),
                    number=number_repeat,
                )
                / number_repeat
            )

        print("Average duration per function:")
        for key, val in dict_timings.items():
            print(f"  {key} : {np.round(val*1000, 2)}ms")

        print(
            "fft, sw, bandpass, coherence and stft are timings for an individual channel"
        )
