import timeit
import numpy as np
from py_neuromodulation import (
    nm_normalization,
    nm_stft,
    nm_fft,
    nm_notch_filter,
    nm_bandpower,
    nm_filter
)

class NM_Timer:

    def __init__(self, analyzer) -> None:
        self.analyzer = analyzer

        self.get_timings()

    def get_timings(self, number_repeat=1000):

        features_ = {}
        ch_idx = 0
        ch = "ECOG_L_1_SMC_AT"
        data = np.random.random([15, 1375])
        
        dict_timings = {}

        dict_timings["time_rereference"] = timeit.timeit(
             lambda: self.analyzer.reference.rereference(data),
             number=number_repeat
        ) / number_repeat


        data = np.random.random([11, 1375])

        dict_timings["time_resample"] = timeit.timeit(
             lambda: self.analyzer.resample.raw_resampling(data),
             number=number_repeat
        ) / number_repeat
        
        data = np.random.random([11, 1000])
        NUM_CH = 11

        dict_timings["time_notchfilter"] = timeit.timeit(
             lambda: nm_notch_filter.notch_filter(
                data,
                self.analyzer.fs,
                self.analyzer.line_noise
            ),
             number=number_repeat
        ) / number_repeat

        dict_timings["time_norm_raw"] = timeit.timeit(
            lambda: nm_normalization.normalize_raw(
                data,
                int(
                self.analyzer.settings["raw_normalization_settings"]["normalization_time"] * self.analyzer.fs
                ),
                self.analyzer.fs,
                self.analyzer.settings["raw_normalization_settings"]["normalization_method"],
                self.analyzer.settings["raw_normalization_settings"]["clip"]),
                number=number_repeat
        ) / number_repeat

        feature_series = self.analyzer.feature_arr_raw.iloc[-1]
        dict_timings["time_feature_norm"] = timeit.timeit(
            lambda: nm_normalization.normalize_features(
                    feature_series, self.analyzer.feature_arr_raw,
                    self.analyzer.feat_normalize_samples,
                    self.analyzer.settings["feature_normalization_settings"][
                        "normalization_method"],
                    self.analyzer.settings["feature_normalization_settings"][
                        "clip"]),
                number=number_repeat
        ) / number_repeat

        dict_timings["time_projection"] = timeit.timeit(
            lambda: self.analyzer.next_projection_run(feature_series),
                number=number_repeat
        ) / number_repeat

        dict_timings["time_applyfilterband"] = timeit.timeit(
            lambda: nm_filter.apply_filter(data, self.analyzer.features.filter_fun),
                number=number_repeat
        ) / number_repeat

        dict_timings["time_sw"] = timeit.timeit(
            lambda: self.analyzer.features.sw_features.get_sharpwave_features(
                features_, data[ch_idx, -100:], ch),
                number=number_repeat
        ) / number_repeat

        dict_timings["time_stft"] = timeit.timeit(
            lambda: nm_stft.get_stft_features(
                features_,
                self.analyzer.features.s,
                self.analyzer.features.fs,
                data[ch_idx, :],
                self.analyzer.features.KF_dict,
                ch,
                self.analyzer.features.f_ranges,
                self.analyzer.features.fband_names),
                number=number_repeat
        ) / number_repeat

        dict_timings["time_fft"] = timeit.timeit(
            lambda: nm_fft.get_fft_features(
                features_,
                self.analyzer.features.s,
                self.analyzer.features.fs,
                data[ch_idx, :],
                self.analyzer.features.KF_dict,
                ch,
                self.analyzer.features.f_ranges, self.analyzer.features.fband_names),
                number=number_repeat
        ) / number_repeat

        seglengths = np.floor(
                self.analyzer.fs / 1000 * np.array([value for value in self.analyzer.features.s[
                        "bandpass_filter_settings"]["segment_lengths"].values()])).astype(int)


        dat_filtered = nm_filter.apply_filter(data, self.analyzer.features.filter_fun)  # shape (bands, time)
        dict_timings["time_bandpass_filter"] = timeit.timeit(
            lambda: nm_bandpower.get_bandpower_features(
                features_,
                self.analyzer.features.s,
                seglengths,
                dat_filtered,
                self.analyzer.features.KF_dict,
                ch, ch_idx
                ),
                number=number_repeat
        ) / number_repeat

        coh_obj = self.analyzer.features.coherence_objects[0]
        dict_timings["time_coherence"] = timeit.timeit(
            lambda: coh_obj.get_coh(
                features_,
                data[coh_obj.ch_1_idx, :],
                data[coh_obj.ch_2_idx, :]
            ),
            number=number_repeat
        ) / number_repeat

        for key, val in dict_timings.items():
            if key == "time_sw":
                print("per channel:\n")
            print(f"{key} : {np.round(val*1000, 2)}ms")
