import numpy as np
from collections.abc import Iterable


from typing import TYPE_CHECKING, Annotated
from pydantic import Field, field_validator

from py_neuromodulation.utils.types import (
    NMFeature,
    BoolSelector,
    FrequencyRange,
    NMBaseModel,
)
from py_neuromodulation.utils import logger

if TYPE_CHECKING:
    from py_neuromodulation import NMSettings


class CoherenceMethods(BoolSelector):
    coh: bool = True
    icoh: bool = True


class CoherenceFeatures(BoolSelector):
    mean_fband: bool = True
    max_fband: bool = True
    max_allfbands: bool = True


ListOfTwoStr = Annotated[list[str], Field(min_length=2, max_length=2)]


class CoherenceSettings(NMBaseModel):
    features: CoherenceFeatures = CoherenceFeatures()
    method: CoherenceMethods = CoherenceMethods()
    channels: list[ListOfTwoStr] = []
    frequency_bands: list[str] = Field(default=["high_beta"], min_length=1)

    @field_validator("frequency_bands")
    def fbands_spaces_to_underscores(cls, frequency_bands):
        return [f.replace(" ", "_") for f in frequency_bands]


class CoherenceObject:
    def __init__(
        self,
        sfreq: float,
        window: str,
        fbands: list[FrequencyRange],
        fband_names: list[str],
        ch_1_name: str,
        ch_2_name: str,
        ch_1_idx: int,
        ch_2_idx: int,
        coh: bool,
        icoh: bool,
        features_coh: CoherenceFeatures,
    ) -> None:
        self.sfreq = sfreq
        self.window = window
        self.fbands = fbands
        self.fband_names = fband_names
        self.ch_1 = ch_1_name
        self.ch_2 = ch_2_name
        self.ch_1_idx = ch_1_idx
        self.ch_2_idx = ch_2_idx
        self.coh = coh
        self.icoh = icoh
        self.features_coh = features_coh

        self.Pxx = None
        self.Pyy = None
        self.Pxy = None
        self.f = None
        self.coh_val = None
        self.icoh_val = None

    def get_coh(self, feature_results, x, y):
        from scipy.signal import welch, csd

        self.f, self.Pxx = welch(x, self.sfreq, self.window, nperseg=128)
        self.Pyy = welch(y, self.sfreq, self.window, nperseg=128)[1]
        self.Pxy = csd(x, y, self.sfreq, self.window, nperseg=128)[1]

        if self.coh:
            self.coh_val = np.abs(self.Pxy**2) / (self.Pxx * self.Pyy)
        if self.icoh:
            self.icoh_val = np.array(self.Pxy / (self.Pxx * self.Pyy)).imag

        for coh_idx, coh_type in enumerate([self.coh, self.icoh]):
            if coh_type:
                if coh_idx == 0:
                    coh_val = self.coh_val
                    coh_name = "coh"
                else:
                    coh_val = self.icoh_val
                    coh_name = "icoh"

            for idx, fband in enumerate(self.fbands):
                if self.features_coh.mean_fband:
                    feature_calc = np.mean(
                        coh_val[np.bitwise_and(self.f > fband[0], self.f < fband[1])]
                    )
                    feature_name = "_".join(
                        [
                            coh_name,
                            self.ch_1,
                            "to",
                            self.ch_2,
                            "mean_fband",
                            self.fband_names[idx],
                        ]
                    )
                    feature_results[feature_name] = feature_calc
                if self.features_coh.max_fband:
                    feature_calc = np.max(
                        coh_val[np.bitwise_and(self.f > fband[0], self.f < fband[1])]
                    )
                    feature_name = "_".join(
                        [
                            coh_name,
                            self.ch_1,
                            "to",
                            self.ch_2,
                            "max_fband",
                            self.fband_names[idx],
                        ]
                    )
                    feature_results[feature_name] = feature_calc
            if self.features_coh.max_allfbands:
                feature_calc = self.f[np.argmax(coh_val)]
                feature_name = "_".join(
                    [
                        coh_name,
                        self.ch_1,
                        "to",
                        self.ch_2,
                        "max_allfbands",
                        self.fband_names[idx],
                    ]
                )
                feature_results[feature_name] = feature_calc
        return feature_results


class Coherence(NMFeature):
    def __init__(
        self, settings: "NMSettings", ch_names: list[str], sfreq: float
    ) -> None:
        self.settings = settings.coherence_settings
        self.frequency_ranges_hz = settings.frequency_ranges_hz
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.coherence_objects: Iterable[CoherenceObject] = []

        self.test_settings(settings, ch_names, sfreq)

        for idx_coh in range(len(self.settings.channels)):
            fband_names = self.settings.frequency_bands
            fband_specs = []
            for band_name in fband_names:
                fband_specs.append(self.frequency_ranges_hz[band_name])

            ch_1_name = self.settings.channels[idx_coh][0]
            ch_1_name_reref = [ch for ch in self.ch_names if ch.startswith(ch_1_name)][
                0
            ]
            ch_1_idx = self.ch_names.index(ch_1_name_reref)

            ch_2_name = self.settings.channels[idx_coh][1]
            ch_2_name_reref = [ch for ch in self.ch_names if ch.startswith(ch_2_name)][
                0
            ]
            ch_2_idx = self.ch_names.index(ch_2_name_reref)

            self.coherence_objects.append(
                CoherenceObject(
                    sfreq,
                    "hann",
                    fband_specs,
                    fband_names,
                    ch_1_name,
                    ch_2_name,
                    ch_1_idx,
                    ch_2_idx,
                    self.settings.method.coh,
                    self.settings.method.icoh,
                    self.settings.features,
                )
            )

    @staticmethod
    def test_settings(
        settings: "NMSettings",
        ch_names: Iterable[str],
        sfreq: float,
    ):
        flat_channels = [
            ch for ch_pair in settings.coherence_settings.channels for ch in ch_pair
        ]

        valid_coh_channel = [
            sum(ch.startswith(ch_coh) for ch in ch_names) for ch_coh in flat_channels
        ]
        for ch_idx, ch_coh in enumerate(flat_channels):
            if valid_coh_channel[ch_idx] == 0:
                raise RuntimeError(
                    f"Coherence selected channel {ch_coh} does not match any channel name: \n"
                    f"  - settings.coherence_settings.channels: {settings.coherence_settings.channels}\n"
                    f"  - ch_names: {ch_names} \n"
                )

            if valid_coh_channel[ch_idx] > 1:
                raise RuntimeError(
                    f"Coherence selected channel {ch_coh} is ambigous and matches more than one channel name: \n"
                    f"  - settings.coherence_settings.channels: {settings.coherence_settings.channels}\n"
                    f"  - ch_names: {ch_names} \n"
                )

        assert all(
            f_band_coh in settings.frequency_ranges_hz
            for f_band_coh in settings.coherence_settings.frequency_bands
        ), (
            "coherence selected frequency bands don't match the ones"
            "specified in s['frequency_ranges_hz']"
            f"coherence frequency bands: {settings.coherence_settings.frequency_bands}"
            f"specified frequency_ranges_hz: {settings.frequency_ranges_hz}"
        )

        assert all(
            settings.frequency_ranges_hz[fb][0] < sfreq / 2
            and settings.frequency_ranges_hz[fb][1] < sfreq / 2
            for fb in settings.coherence_settings.frequency_bands
        ), (
            "the coherence frequency band ranges need to be smaller than the Nyquist frequency"
            f"got sfreq = {sfreq} and fband ranges {settings.coherence_settings.frequency_bands}"
        )

        if not settings.coherence_settings.method.get_enabled():
            logger.warn(
                "feature coherence enabled, but no coherence['method'] selected"
            )

    def calc_feature(self, data: np.ndarray) -> dict:
        feature_results = {}

        for coh_obj in self.coherence_objects:
            feature_results = coh_obj.get_coh(
                feature_results,
                data[coh_obj.ch_1_idx, :],
                data[coh_obj.ch_2_idx, :],
            )

        return feature_results
