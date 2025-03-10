from collections.abc import Iterable
import numpy as np

from typing import TYPE_CHECKING, Annotated, Literal
from pydantic import Field

from py_neuromodulation.utils.types import NMFeature, NMBaseModel
from py_neuromodulation.utils.pydantic_extensions import NMField

if TYPE_CHECKING:
    from py_neuromodulation import NMSettings


ListOfTwoStr = Annotated[list[str], Field(min_length=2, max_length=2)]


MNE_CONNECTIVITY_METHOD = Literal[
    "coh",
    "cohy",
    "imcoh",
    "cacoh",
    "mic",
    "mim",
    "plv",
    "ciplv",
    "ppc",
    "pli",
    "dpli",
    "wpli",
    "wpli2_debiased",
    "gc",
    "gc_tr",
]

MNE_CONNECTIVITY_MODE = Literal["multitaper", "fourier", "cwt_morlet"]


class MNEConnectivitySettings(NMBaseModel):
    method: MNE_CONNECTIVITY_METHOD = NMField(default="plv")
    mode: MNE_CONNECTIVITY_MODE = NMField(default="multitaper")
    channels: list[ListOfTwoStr] = []


class MNEConnectivity(NMFeature):
    def __init__(
        self,
        settings: "NMSettings",
        ch_names: Iterable[str],
        sfreq: float,
    ) -> None:
        self.settings = settings

        self.ch_names = ch_names
        self.sfreq = sfreq

        self.channels = settings.mne_connectivity_settings.channels

        # Params used by spectral_connectivity_epochs
        self.mode = settings.mne_connectivity_settings.mode
        self.method = settings.mne_connectivity_settings.method
        self.indices = ([], [])  # convert channel names to channel indices in data
        for con_idx in range(len(self.channels)):
            seed_name = self.channels[con_idx][0]
            target_name = self.channels[con_idx][1]
            seed_name_reref = [ch for ch in self.ch_names if ch.startswith(seed_name)][0]
            target_name_reref = [ch for ch in self.ch_names if ch.startswith(target_name)][0]
            self.indices[0].append(self.ch_names.index(seed_name_reref))
            self.indices[1].append(self.ch_names.index(target_name_reref))

        self.fbands = settings.frequency_ranges_hz
        self.fband_ranges: list = []
        self.result_keys = []

        self.prev_batch_shape: tuple = (-1, -1)  # sentinel value

    def calc_feature(self, data: np.ndarray) -> dict:
        from mne_connectivity import spectral_connectivity_epochs

        # n_jobs is here kept to 1, since setup of the multiprocessing Pool
        # takes longer than most batch computing sizes
        spec_out = spectral_connectivity_epochs(
            data=np.expand_dims(data, axis=0),  # add singleton epoch dimension
            sfreq=self.sfreq,
            method=self.method,
            mode=self.mode,
            indices=self.indices,
            verbose=False,
        )
        dat_conn: np.ndarray = spec_out.get_data()

        # Get frequency band ranges only for the first batch, it's already the same
        if len(self.fband_ranges) == 0:
            for fband_range in self.fbands.values():
                self.fband_ranges.append(
                    np.where(
                        (np.array(spec_out.freqs) >= fband_range[0])
                        & (np.array(spec_out.freqs) <= fband_range[1])
                    )[0]
                )

        feature_results = {}
        for con_idx in np.arange(dat_conn.shape[0]):
            for fband_idx, fband_name in enumerate(self.fbands):
                # TODO: Add support for max_fband and max_allfbands
                feature_results[
                    "_".join(
                        [
                            self.method,
                            self.channels[con_idx][0],  # seed channel name
                            "to",
                            self.channels[con_idx][1],  # target channel name
                            "mean_fband",
                            fband_name,
                        ]
                    )
                ] = np.mean(dat_conn[con_idx, self.fband_ranges[fband_idx]])

        # Store current experiment parameters to check if re-initialization is needed
        self.prev_batch_shape = data.shape

        return feature_results
