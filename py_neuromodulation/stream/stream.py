"""Module for generic and offline data streams."""

import asyncio
from typing import TYPE_CHECKING
from collections.abc import Iterator
import numpy as np

import multiprocessing as mp
from contextlib import suppress

from py_neuromodulation.features import USE_FREQ_RANGES
from py_neuromodulation.utils.types import _PathLike
from py_neuromodulation.utils import logger
from py_neuromodulation.utils.data_writer import DataWriter
from py_neuromodulation.gui.backend.app_socket import WebSocketManager
from py_neuromodulation.stream.rawdata_generator import RawDataGenerator
from py_neuromodulation.stream.data_processor import DataProcessor

if TYPE_CHECKING:
    import pandas as pd


class Stream:
    """_GenericStream base class.
    This class can be inherited for different types of offline streams

    Parameters
    ----------
    nm_stream_abc : stream_abc.NMStream
    """

    def __init__(
        self,
        verbose: bool = True,
    ) -> None:
        self.verbose = verbose
        self.is_running = False
        

    async def run(
        self,
        data_processor: DataProcessor | None = None,
        data_generator : Iterator | None = None,
        data_writer: DataWriter | None = None,
        stream_handling_queue: asyncio.Queue | None = None,
        websocket_featues: WebSocketManager | None = None,
    ):
        self.data_processor = data_processor
        # Check that at least one channel is selected for analysis
        if self.data_processor.channels.query("used == 1 and target == 0").shape[0] == 0:
            raise ValueError(
                "No channels selected for analysis that have column 'used' = 1 and 'target' = 0. Please check your channels"
            )

        # If features that use frequency ranges are on, test them against nyquist frequency
        need_nyquist_check = any(
            (f in USE_FREQ_RANGES for f in self.data_processor.settings.features.get_enabled())
        )

        if need_nyquist_check:
            assert all(
                fb.frequency_high_hz < self.data_processor.sfreq_raw / 2
                for fb in self.data_processor.settings.frequency_ranges_hz.values()
            ), (
                "If a feature that uses frequency ranges is selected, "
                "the frequency band ranges need to be smaller than the nyquist frequency.\n"
                f"Got sfreq = {self.data_processor.sfreq_raw} and fband ranges:\n {self.data_processor.settings.frequency_ranges_hz}"
            )

        self.stream_handling_queue = stream_handling_queue
        self.is_running = False
        self.is_lslstream = type(data_generator) != RawDataGenerator

        prev_batch_end = 0
        for timestamps, data_batch in data_generator:
            self.is_running = True
            if self.stream_handling_queue is not None:
                await asyncio.sleep(0.001)
                if not self.stream_handling_queue.empty():
                    stop_signal = await asyncio.wait_for(self.stream_handling_queue.get(), timeout=0.01)
                    if stop_signal == "stop":
                        break
            if data_batch is None:
                break

            feature_dict = data_processor.process(data_batch)

            this_batch_end = timestamps[-1]
            batch_length = this_batch_end - prev_batch_end
            logger.debug(
                f"{batch_length:.3f} seconds of new data processed",
            )

            feature_dict["time"] = (
                batch_length
                if self.is_lslstream
                else np.ceil(this_batch_end * 1000 + 1)
            )

            prev_batch_end = this_batch_end

            if self.verbose:
                logger.info("Time: %.2f", feature_dict["time"] / 1000)

            feature_dict = data_generator.add_target(feature_dict, data_batch)

            with suppress(TypeError):  # Need this because some features output None
                for key, value in feature_dict.items():
                    feature_dict[key] = np.float64(value)

            data_writer.write_data(feature_dict)

            if websocket_featues is not None:
                await websocket_featues.send_cbor(feature_dict)

        feature_df = data_writer.get_features()
        data_writer.save_csv_features(feature_df)
        self._save_sidecars_after_stream(data_writer.out_dir, data_writer.experiment_name)
        self.is_running = False

        return feature_df

    def plot_raw_signal(
        self,
        sfreq: float | None = None,
        data: np.ndarray | None = None,
        lowpass: float | None = None,
        highpass: float | None = None,
        picks: list | None = None,
        plot_time: bool = True,
        plot_psd: bool = False,
    ) -> None:
        """Use MNE-RawArray Plot to investigate PSD or raw_signal plot.

        Parameters
        ----------
        sfreq : float
            sampling frequency [Hz]
        data : np.ndarray, optional
            data (n_channels, n_times), by default None
        lowpass: float, optional
            cutoff lowpass filter frequency
        highpass: float, optional
            cutoff highpass filter frequency
        picks: list, optional
            list of channels to plot
        plot_time : bool, optional
            mne.io.RawArray.plot(), by default True
        plot_psd : bool, optional
            mne.io.RawArray.plot(), by default False

        Raises
        ------
        ValueError
            raise Exception when no data is passed
        """
        if self.data is None and data is None:
            raise ValueError("No data passed to plot_raw_signal function.")

        if data is None and self.data is not None:
            data = self.data

        if sfreq is None:
            sfreq = self.sfreq

        if self.channels is not None:
            ch_names = self.channels["name"].to_list()
            ch_types = self.channels["type"].to_list()
        else:
            ch_names = [f"ch_{i}" for i in range(data.shape[0])]
            ch_types = ["ecog" for i in range(data.shape[0])]

        from mne import create_info
        from mne.io import RawArray

        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = RawArray(data, info)

        if picks is not None:
            raw = raw.pick(picks)
        self.raw = raw
        if plot_time:
            raw.plot(highpass=highpass, lowpass=lowpass)
        if plot_psd:
            raw.compute_psd().plot()

    def _save_sidecars_after_stream(
        self,
        out_dir: _PathLike,
        experiment_name: str = "experiment"
    ) -> None:
        """Save settings, nm_channels and sidecar after run"""
        self._save_sidecar(out_dir, experiment_name)
        self._save_settings(out_dir, experiment_name)
        self._save_channels(out_dir, experiment_name)

    def _save_channels(self, out_dir, experiment_name) -> None:
        self.data_processor.save_channels(out_dir, experiment_name)

    def _save_settings(self, out_dir, experiment_name) -> None:
        self.data_processor.save_settings(out_dir, experiment_name)

    def _save_sidecar(self, out_dir, experiment_name) -> None:
        """Save sidecar incduing fs, coords, sess_right to
        out_path_root and subfolder 'folder_name'"""
        self.data_processor.save_sidecar(
            out_dir, experiment_name
        )
