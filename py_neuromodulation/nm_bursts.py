import numpy as np
from collections.abc import Iterable
from scipy.signal import hilbert as scipy_hilbert

from py_neuromodulation.nm_features_abc import Feature
from py_neuromodulation.nm_filter import MNEFilter


class Burst(Feature):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.s = settings
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.threshold = self.s["burst_settings"]["threshold"]
        self.time_duration_s = self.s["burst_settings"]["time_duration_s"]
        self.samples_overlap = int(
            self.sfreq
            * (self.s["segment_length_features_ms"] / 1000)
            / self.s["sampling_rate_features_hz"]
        )

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

        self.num_max_samples_ring_buffer = int(
            self.sfreq * self.time_duration_s
        )

        self.bandpass_filter = MNEFilter(
            f_ranges=self.f_ranges,
            sfreq=self.sfreq,
            filter_length=self.sfreq - 1,
            verbose=False,
        )

        # create dict with fband, channel specific data store
        # for previous time_duration_s
        def init_ch_fband_dict() -> dict:
            d : dict = {}
            for ch in self.ch_names:
                d.setdefault(ch , {})
                for fb in self.fband_names:
                    d[ch].setdefault(fb, None)

            return d

        self.data_buffer = init_ch_fband_dict()
        
    @staticmethod
    def test_settings(
        settings: dict,
        ch_names: Iterable[str],
        sfreq: float,
    ):
        assert isinstance(
            settings["burst_settings"]["threshold"], (float, int)
        ), f"burst settings threshold needs to be type int or float, got: {settings['burst_settings']['threshold']}"
        assert (
            0 < settings["burst_settings"]["threshold"] < 100
        ), f"burst setting threshold needs to be between 0 and 100, got: {settings['burst_settings']['threshold']}"
        assert isinstance(
            settings["burst_settings"]["time_duration_s"], (float, int)
        ), f"burst settings time_duration_s needs to be type int or float, got: {settings['burst_settings']['time_duration_s']}"
        assert (
            settings["burst_settings"]["time_duration_s"] > 0
        ), f"burst setting time_duration_s needs to be greater than 0, got: {settings['burst_settings']['time_duration_s']}"

        for fband_burst in settings["burst_settings"]["frequency_bands"]:
            assert fband_burst in list(
                settings["frequency_ranges_hz"].keys()
            ), f"bursting {fband_burst} needs to be defined in settings['frequency_ranges_hz']"

        for burst_feature in settings["burst_settings"][
            "burst_features"
        ].keys():
            assert isinstance(
                settings["burst_settings"]["burst_features"][burst_feature],
                bool,
            ), (
                f"bursting feature {burst_feature} needs to be type bool, "
                f"got: {settings['burst_settings']['burst_features'][burst_feature]}"
            )

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        # filter_data returns (n_channels, n_fbands, n_samples)
        filtered_data = np.abs(
            scipy_hilbert(self.bandpass_filter.filter_data(data), axis=2)
        )
        for ch_idx, ch_name in enumerate(self.ch_names):
            for fband_idx, fband_name in enumerate(self.fband_names):
                new_dat = filtered_data[ch_idx, fband_idx, :]
                if self.data_buffer[ch_name][fband_name] is None:
                    self.data_buffer[ch_name][fband_name] = new_dat
                else:
                    self.data_buffer[ch_name][fband_name] = np.concatenate(
                        (
                            self.data_buffer[ch_name][fband_name],
                            new_dat[-self.samples_overlap :],
                        ),
                        axis=0,
                    )[-self.num_max_samples_ring_buffer :]

                # calc features
                burst_thr : float = np.percentile(
                    self.data_buffer[ch_name][fband_name], q=self.threshold
                )

                burst_amplitude, burst_length = self.get_burst_amplitude_length(
                    new_dat, burst_thr, self.sfreq
                )

                features_compute[
                    f"{ch_name}_bursts_{fband_name}_duration_mean"
                ] = (np.mean(burst_length) if len(burst_length) != 0 else 0)
                features_compute[
                    f"{ch_name}_bursts_{fband_name}_amplitude_mean"
                ] = (
                    np.mean([np.mean(a) for a in burst_amplitude])
                    if len(burst_length) != 0
                    else 0
                )

                features_compute[
                    f"{ch_name}_bursts_{fband_name}_duration_max"
                ] = (np.max(burst_length) if len(burst_length) != 0 else 0)
                features_compute[
                    f"{ch_name}_bursts_{fband_name}_amplitude_max"
                ] = (
                    np.max([np.max(a) for a in burst_amplitude])
                    if len(burst_amplitude) != 0
                    else 0
                )

                features_compute[
                    f"{ch_name}_bursts_{fband_name}_burst_rate_per_s"
                ] = (
                    np.mean(burst_length)
                    / (self.s["segment_length_features_ms"] / 1000)
                    if len(burst_length) != 0
                    else 0
                )

                in_burst = False
                if self.data_buffer[ch_name][fband_name][-1] > burst_thr:
                    in_burst = True

                features_compute[f"{ch_name}_bursts_{fband_name}_in_burst"] = (
                    in_burst
                )
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
        burst_length = []
        burst_amplitude = []

        burst_time_points = np.where(deriv)[0]

        for i in range(burst_time_points.size // 2):
            burst_length.append(
                burst_time_points[2 * i + 1] - burst_time_points[2 * i]
            )
            burst_amplitude.append(
                beta_averp_norm[
                    burst_time_points[2 * i] : burst_time_points[2 * i + 1]
                ]
            )

        # the last burst length (in case isburst == True) is omitted,
        # since the true burst length cannot be estimated
        return burst_amplitude, np.array(burst_length) / sfreq

