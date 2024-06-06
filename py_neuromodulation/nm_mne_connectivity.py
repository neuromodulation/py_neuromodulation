from collections.abc import Iterable
import numpy as np
from typing import TYPE_CHECKING

from py_neuromodulation.nm_features import NMFeature
from py_neuromodulation.nm_types import NMBaseModel

if TYPE_CHECKING:
    from py_neuromodulation.nm_settings import NMSettings
    from mne.io import RawArray
    from mne import Epochs


class MNEConnectivitySettings(NMBaseModel):
    method: str = "plv"
    mode: str = "multitaper"


class MNEConnectivity(NMFeature):
    def __init__(
        self,
        settings: "NMSettings",
        ch_names: Iterable[str],
        sfreq: float,
    ) -> None:
        self.ch_names = ch_names
        self.settings = settings
        self.mode = settings.mne_connectivity.mode
        self.method = settings.mne_connectivity.method
        self.sfreq = sfreq

        self.fbands = settings.frequency_ranges_hz
        self.fband_ranges: list = []

    @staticmethod
    def get_epoched_data(raw: "RawArray", epoch_length: float = 1) -> "Epochs":
        time_samples_s = raw.get_data().shape[1] / raw.info["sfreq"]
        if epoch_length > time_samples_s:
            raise ValueError(
                f"the intended epoch length for mne connectivity: {epoch_length}s"
                f" are longer than the passed data array {np.round(time_samples_s, 2)}s"
            )

        from mne import make_fixed_length_events, Epochs

        events = make_fixed_length_events(raw, duration=epoch_length, overlap=0)
        event_id = {"rest": 1}

        epochs = Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=0,
            tmax=epoch_length,
            baseline=None,
            reject_by_annotation=True,
        )
        if epochs.events.shape[0] < 2:
            raise Exception(
                f"A minimum of 2 epochs is required for mne_connectivity,"
                f" got only {epochs.events.shape[0]}. Increase settings['segment_length_features_ms']"
            )
        return epochs

    def estimate_connectivity(self, epochs: "Epochs"):
        # n_jobs is here kept to 1, since setup of the multiprocessing Pool
        # takes longer than most batch computing sizes
        from mne_connectivity import spectral_connectivity_epochs

        spec_out = spectral_connectivity_epochs(
            data=epochs,
            sfreq=self.sfreq,
            n_jobs=1,
            method=self.method,
            mode=self.mode,
            indices=(np.array([0, 0, 1, 1]), np.array([2, 3, 2, 3])),
            faverage=False,
            block_size=1000,
        )
        return spec_out

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        from mne.io import RawArray
        from mne import create_info

        raw = RawArray(
            data=data,
            info=create_info(ch_names=self.ch_names, sfreq=self.sfreq),
        )
        epochs = self.get_epoched_data(raw)
        # there need to be minimum 2 of two epochs, otherwise mne_connectivity
        # is not correctly initialized

        spec_out = self.estimate_connectivity(epochs)

        if not self.fband_ranges:
            for fband_name, fband_range in self.fbands:
                self.fband_ranges.append(
                    np.where(
                        np.logical_and(
                            np.array(spec_out.freqs) > fband_range[0],
                            np.array(spec_out.freqs) < fband_range[1],
                        )
                    )[0]
                )

        dat_conn = spec_out.get_data()
        for conn in np.arange(dat_conn.shape[0]):
            for fband_idx, fband in enumerate(self.fbands):
                features_compute["_".join(["ch1", self.method, str(conn), fband])] = (
                    np.mean(dat_conn[conn, self.fband_ranges[fband_idx]])
                )

        return features_compute
