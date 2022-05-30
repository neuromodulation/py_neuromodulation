import enum
import numpy as np
from typing import Iterable

from pkg_resources import get_build_platform

from py_neuromodulation import nm_features_abc, nm_oscillatory, nm_filter


class Burst(nm_features_abc.Feature):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float
    ) -> None:

        self.s = settings
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.threshold = self.s["burst_settings"]["threshold"]
        self.time_duration_s = self.s["burst_settings"]["time_duration_s"]

        self.fband_names = self.s["burst_settings"]["frequency_bands"]
        self.f_ranges = [
            self.s["frequency_ranges_hz"][fband_name]
            for fband_name in self.fband_names
        ]
        self.seglengths = np.floor(
            self.sfreq
            / 1000
            * np.array(
                [
                    self.s["bandpass_filter_settings"]["segment_lengths_ms"][
                        fband
                    ]
                    for fband in self.fband_names
                ]
            )
        ).astype(int)

        self.number_samples_calc_features = int(
            self.sfreq / self.s["sampling_rate_features_hz"]
        )

        self.bandpass_filter = nm_filter.BandPassFilter(
            f_ranges=self.f_ranges,
            sfreq=self.sfreq,
            filter_length=self.sfreq - 1,
            verbose=False,
        )

        # create dict with fband, channel specific data store
        # for previous time_duration_s
        def init_ch_fband_dict() -> dict:
            d = {}
            for ch in self.ch_names:
                if ch not in d:
                    d[ch] = {}
                for fb in self.fband_names:
                    if fb not in d[ch]:
                        d[ch][fb] = None
            return d

        self.data_buffer = init_ch_fband_dict()

    def calc_feature(self, data: np.array, features_compute: dict) -> dict:

        # filter_data returns (n_channels, n_fbands, n_samples)
        filtered_data = self.bandpass_filter.filter_data(data)
        for ch_idx, ch_name in enumerate(self.ch_names):
            for fband_idx, fband_name in enumerate(self.fband_names):
                new_dat = filtered_data[ch_idx, fband_idx, :]
                if self.data_buffer[ch_name][fband_name] is None:
                    self.data_buffer[ch_name][fband_name] = new_dat
                else:
                    self.data_buffer[ch_name][fband_name] = np.concatenate(
                        (self.data_buffer[ch_name][fband_name], new_dat), axis=0
                    )

                # calc features
                burst_thr = np.percentile(
                    self.data_buffer[ch_name][fband_name], q=self.threshold
                )

                burst_amplitude, burst_length = self.get_burst_amplitude_length(
                    new_dat, burst_thr, self.sfreq
                )

                features_compute[
                    f"bursts_{ch_name}_{fband_name}_duration"
                ] = np.mean(burst_length)
                features_compute[
                    f"bursts_{ch_name}_{fband_name}_amplitude"
                ] = np.mean([np.mean(a) for a in burst_amplitude])
                features_compute[
                    f"bursts_{ch_name}_{fband_name}_burst_rate_per_s"
                ] = np.mean(burst_length) / (
                    self.s["segment_length_features_ms"] / 1000
                )

                in_burst = False
                if self.data_buffer[ch_name][fband_name][-1] > burst_thr:
                    in_burst = True

                features_compute[
                    f"bursts_{ch_name}_{fband_name}_in_burst"
                ] = in_burst
        return features_compute

    @staticmethod
    def get_burst_amplitude_length(
        beta_averp_norm, burst_thr: float, sfreq: float
    ):
        """
        Analysing the duration of beta burst
        """
        bursts = np.zeros((beta_averp_norm.shape[0] + 1), dtype=bool)
        bursts[1:] = beta_averp_norm >= burst_thr
        deriv = np.diff(bursts)
        isburst = False
        burst_length = []
        burst_amplitude = []
        burst_start = 0

        for index, i in enumerate(deriv):
            if i == True:
                if isburst == True:
                    burst_length.append(index - burst_start)
                    burst_amplitude.append(beta_averp_norm[burst_start:index])

                    isburst = False
                else:
                    burst_start = index
                    isburst = True
        if isburst:
            burst_length.append(index + 1 - burst_start)
        burst_length = np.array(burst_length) / sfreq

        return burst_amplitude, burst_length
