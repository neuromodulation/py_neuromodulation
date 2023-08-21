from typing import Iterable
import numpy as np

import mne
from mne_connectivity import spectral_connectivity_epochs

from py_neuromodulation import nm_features_abc


class MNEConnectivity(nm_features_abc.Feature):
    def __init__(
        self,
        settings: dict,
        ch_names: Iterable[str],
        sfreq: float,
    ) -> None:
        self.s = settings
        self.ch_names = ch_names
        self.mode = settings["mne_connectiviy"]["mode"]
        self.method = settings["mne_connectiviy"]["method"]
        self.sfreq = sfreq

        self.fbands = list(self.s["frequency_ranges_hz"].keys())
        self.fband_ranges = []

    @staticmethod
    def test_settings(
        settings: dict,
        ch_names: Iterable[str],
        sfreq: int | float,
    ):
        # TODO: Double check passed parameters with mne_connectivity
        pass

    @staticmethod
    def get_epoched_data(
        raw: mne.io.RawArray, epoch_length: float = 1
    ) -> np.array:
        time_samples_s = raw.get_data().shape[1] / raw.info["sfreq"]
        if epoch_length > time_samples_s:
             raise ValueError(
                f"the intended epoch length for mne connectivity: {epoch_length}s"
                f" are longer than the passed data array {np.round(time_samples_s, 2)}s"
             )
        events = mne.make_fixed_length_events(
            raw, duration=epoch_length, overlap=0
        )
        event_id = {"rest": 1}

        epochs = mne.Epochs(
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
                f" got only {epochs.events.shape[0]}. Increase settings['segment_length']"
            )
        return epochs

    def estimate_connectivity(self, epochs: mne.Epochs):
        # n_jobs is here kept to 1, since setup of the multiprocessing Pool 
        # takes longer than most batch computing sizes

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

    def calc_feature(self, data: np.array, features_compute: dict) -> dict:

        raw = mne.io.RawArray(
            data=data,
            info=mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq),
        )
        epochs = self.get_epoched_data(raw)
        # there need to be minimum 2 of two epochs, otherwise mne_connectivity 
        # is not correctly initialized

        spec_out = self.estimate_connectivity(epochs)
        if len(self.fband_ranges) == 0:
            for fband in self.fbands:
                self.fband_ranges.append(
                    np.where(
                        np.logical_and(
                            np.array(spec_out.freqs)
                            > self.s["frequency_ranges_hz"][fband][0],
                            np.array(spec_out.freqs)
                            < self.s["frequency_ranges_hz"][fband][1],
                        )
                    )[0]
                )
        dat_conn = spec_out.get_data()
        for conn in np.arange(dat_conn.shape[0]):
            for fband_idx, fband in enumerate(self.fbands):
                features_compute[
                    "_".join(["ch1", self.method, str(conn), fband])
                ] = np.mean(dat_conn[conn, self.fband_ranges[fband_idx]])

        return features_compute
