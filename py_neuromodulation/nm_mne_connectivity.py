from collections.abc import Iterable
import numpy as np
import pandas as pd
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
        from mne import create_info

        self.settings = settings

        self.ch_names = ch_names
        self.sfreq = sfreq

        # Params used by spectral_connectivity_epochs
        self.mode = settings.mne_connectivity.mode
        self.method = settings.mne_connectivity.method

        self.fbands = settings.frequency_ranges_hz
        self.fband_ranges: list = []
        self.result_keys = []

        self.raw_info = create_info(ch_names=self.ch_names, sfreq=self.sfreq)
        self.raw_array: "RawArray"
        self.epochs: "Epochs"
        self.prev_batch_shape: tuple = (-1, -1)  # sentinel value

    def calc_feature(self, data: np.ndarray) -> dict:
        from mne.io import RawArray
        from mne import Epochs
        from mne_connectivity import spectral_connectivity_epochs

        time_samples_s = data.shape[1] / self.sfreq
        epoch_length: float = 1  # TODO: Make this a parameter?

        if epoch_length > time_samples_s:
            raise ValueError(
                f"the intended epoch length for mne connectivity: {epoch_length}s"
                f" are longer than the passed data array {np.round(time_samples_s, 2)}s"
            )

        # Only reinitialize the raw_array and epochs object if the data shape has changed
        # That could mean that the channels have been re-selected, or we're in the last batch
        # TODO: If sfreq or channels change, do we re-initialize the whole Stream object?
        if data.shape != self.prev_batch_shape:
            self.raw_array = RawArray(
                data=data,
                info=self.raw_info,
                copy=None,  # type: ignore
                verbose=False,
            )

            # self.events = make_fixed_length_events(self.raw_array, duration=epoch_length)
            # Equivalent code for those parameters:
            event_times = np.arange(
                0, data.shape[-1], self.sfreq * epoch_length, dtype=int
            )
            events = np.column_stack(
                (
                    event_times,
                    np.zeros_like(event_times, dtype=int),
                    np.ones_like(event_times, dtype=int),
                )
            )

            # there need to be minimum 2 of two epochs, otherwise mne_connectivity
            # is not correctly initialized
            if events.shape[0] < 2:
                raise RuntimeError(
                    f"A minimum of 2 epochs is required for mne_connectivity,"
                    f" got only {events.shape[0]}. Increase settings['segment_length_features_ms']"
                )

            self.epochs = Epochs(
                self.raw_array,
                events=events,
                event_id={"rest": 1},
                tmin=0,
                tmax=epoch_length,
                baseline=None,
                reject_by_annotation=True,
                verbose=False,
            )

            # Trick the function "spectral_connectivity_epochs" into not calling "add_annotations_to_metadata"
            # TODO: This is a hack, and maybe needs a fix in the mne_connectivity library
            self.epochs._metadata = pd.DataFrame(index=np.arange(events.shape[0]))

        else:
            # As long as the initialization parameters, channels, sfreq and batch size are the same
            # We can re-use the existing epochs object by updating the raw data
            self.raw_array._data = data
            self.epochs._raw = self.raw_array

        # n_jobs is here kept to 1, since setup of the multiprocessing Pool
        # takes longer than most batch computing sizes
        spec_out = spectral_connectivity_epochs(
            data=self.epochs,
            sfreq=self.sfreq,
            method=self.method,
            mode=self.mode,
            indices=(np.array([0, 0, 1, 1]), np.array([2, 3, 2, 3])),
            verbose=False,
        )
        dat_conn: np.ndarray = spec_out.get_data()

        # Get frequency band ranges only for the first batch, it's already the same
        if len(self.fband_ranges) == 0:
            for fband_range in self.fbands.values():
                self.fband_ranges.append(
                    np.where(
                        (np.array(spec_out.freqs) > fband_range[0])
                        & (np.array(spec_out.freqs) < fband_range[1])
                    )[0]
                )

        # TODO: If I compute the mean for the entire fband, results are almost the same before
        # normalization (0.9999999... vs 1.0), but some change wildly after normalization (-3 vs 0)
        # Investigate why, is this a bug in normalization?
        feature_results = {}
        for conn in np.arange(dat_conn.shape[0]):
            for fband_idx, fband in enumerate(self.fbands):
                feature_results["_".join(["ch1", self.method, str(conn), fband])] = (
                    np.mean(dat_conn[conn, self.fband_ranges[fband_idx]])
                )

        # Store current experiment parameters to check if re-initialization is needed
        self.prev_batch_shape = data.shape

        return feature_results
