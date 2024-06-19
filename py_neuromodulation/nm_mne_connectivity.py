from collections.abc import Iterable
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

from py_neuromodulation.nm_features import NMFeature
from py_neuromodulation.nm_types import NMBaseModel

if TYPE_CHECKING:
    from py_neuromodulation.nm_settings import NMSettings
    from mne import Epochs
    from mne.io import RawArray


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
        from mne import create_info

        self.ch_names = ch_names
        self.settings = settings
        self.mode = settings.mne_connectivity.mode
        self.method = settings.mne_connectivity.method
        self.sfreq = sfreq

        self.fbands = settings.frequency_ranges_hz
        self.fband_ranges: list = []

        self.raw_info = create_info(ch_names=self.ch_names, sfreq=self.sfreq)

        self.raw_array: "RawArray"
        self.events: np.ndarray
        self.prev_batch_shape: tuple = (-1, -1)  # sentinel value

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        from mne.io import RawArray
        from mne import Epochs
        from mne_connectivity import spectral_connectivity_epochs

        time_samples_s = data.shape[1] / self.sfreq
        epoch_length: float = 1  # TODO: Make this a parameter

        if epoch_length > time_samples_s:
            raise ValueError(
                f"the intended epoch length for mne connectivity: {epoch_length}s"
                f" are longer than the passed data array {np.round(time_samples_s, 2)}s"
            )

        if data.shape != self.prev_batch_shape:
            self.raw_array = RawArray(
                data=data,
                info=self.raw_info,
                copy=None,  # type: ignore
                verbose=False,
            )

            # self.events = make_fixed_length_events(
            #     self.raw_array, duration=epoch_length, overlap=0
            # )
            # Equivalnet code for those parameters:
            self.events = np.column_stack(
                (
                    np.arange(0, data.shape[-1], self.sfreq * epoch_length, dtype=int),
                    np.array([0, 0], dtype=int),
                    np.array([1, 1], dtype=int),
                )
            )

            # there need to be minimum 2 of two epochs, otherwise mne_connectivity
            # is not correctly initialized
            if self.events.shape[0] < 2:
                raise RuntimeError(
                    f"A minimum of 2 epochs is required for mne_connectivity,"
                    f" got only {self.events.shape[0]}. Increase settings['segment_length_features_ms']"
                )

            self.epochs = Epochs(
                self.raw_array,
                events=self.events,
                event_id={"rest": 1},
                tmin=0,
                tmax=epoch_length,
                baseline=None,
                reject_by_annotation=True,
                verbose=False,
            )

            self.epochs.metadata = pd.DataFrame(
                index=np.arange(self.events.shape[0]), columns=["column1", "column2"]
            )

        else:
            # self.raw_array._data = data
            # As long as the initialization parameters, channels, sfreq and batch size are the same
            # We can re-use the existing epochs object by updating the raw data
            self.epochs._raw = self.raw_array

        # n_jobs is here kept to 1, since setup of the multiprocessing Pool
        # takes longer than most batch computing sizes
        spec_out = spectral_connectivity_epochs(
            data=self.epochs,
            sfreq=self.sfreq,
            n_jobs=1,
            method=self.method,
            mode=self.mode,
            indices=(np.array([0, 0, 1, 1]), np.array([2, 3, 2, 3])),
            faverage=False,
            block_size=1000,
            verbose=False,
        )

        if len(self.fband_ranges) == 0:
            for fband_name, fband_range in self.fbands.items():
                self.fband_ranges.append(
                    np.where(
                        (np.array(spec_out.freqs) > fband_range[0])
                        & (np.array(spec_out.freqs) < fband_range[1])
                    )[0]
                )

        dat_conn: np.ndarray = spec_out.get_data()
        for fband_idx, fband in enumerate(self.fbands):
            fband_mean = np.mean(dat_conn[:, self.fband_ranges[fband_idx]], axis=1)
            for conn in np.arange(dat_conn.shape[0]):
                key = "_".join(["ch1", self.method, str(conn), fband])
                features_compute[key] = fband_mean[conn]

        self.prev_batch_shape = data.shape

        return features_compute
