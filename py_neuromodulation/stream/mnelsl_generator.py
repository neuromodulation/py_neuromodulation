from collections.abc import Iterator
import time
from typing import TYPE_CHECKING, Tuple
import numpy as np
from py_neuromodulation.utils import logger
from mne_lsl.lsl import resolve_streams
import os
from .data_generator_abc import DataGeneratorABC

if TYPE_CHECKING:
    from py_neuromodulation import NMSettings


class MNELSLGenerator(DataGeneratorABC):
    """
    Class is used to create and connect to a LSL stream and pull data from it.
    """

    def __init__(self,
                 segment_length_features_ms: float,
                 sampling_rate_features_hz: float,
                 stream_name: str | None = "example_stream",
                 ) -> None:
        """
        Initialize the LSL stream.

        Parameters:
        -----------
        settings : settings.NMSettings object
        stream_name : str, optional
            Name of the stream to connect to. If not provided, the first available stream is used.

        Raises:
        -------
        RuntimeError
            If no stream is running under the provided name or if there are multiple streams running
            under the same name.
        """
        from mne_lsl.stream import StreamLSL

        self.stream: StreamLSL
        # self.keyboard_interrupt = False

        self._n_seconds_wait_before_disconnect = 3
        try:
            if stream_name is None:
                stream_name = resolve_streams()[0].name
                logger.info(
                    f"Stream name not provided. Using first available stream: {stream_name}"
                )
            self.stream = StreamLSL(name=stream_name, bufsize=2).connect(timeout=2)
        except Exception as e:
            msg = f"Could not connect to stream: {e}. Either no stream is running under the name {stream_name} or there is several streams under this name."
            logger.exception(msg)
            raise RuntimeError(msg)

        if self.stream.sinfo is None:
            raise RuntimeError("Stream info is None. Check if the stream is running.")
        else:
            self.sinfo = self.stream.sinfo

        self.winsize = segment_length_features_ms / self.stream.sinfo.sfreq
        self.sampling_interval = 1 / sampling_rate_features_hz
        self.channels = self.get_LSL_channels()
        self.sfreq = self.stream.sinfo.sfreq

    def get_LSL_channels(self) -> "pd.DataFrame":

        from py_neuromodulation.utils import create_channels
        ch_names = self.sinfo.get_channel_names() or [
            "ch" + str(i) for i in range(self.sinfo.n_channels)
        ]
        ch_types = self.sinfo.get_channel_types() or [
            "eeg" for i in range(self.sinfo.n_channels)
        ]
        return create_channels(
            ch_names=ch_names,
            ch_types=ch_types,
            used_types=["eeg", "ecog", "dbs", "seeg"],
        )

    def __iter__(self):
        return self
    
    def __next__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        self.last_time = time.time()
        check_data = None
        data = None
        stream_start_time = None

        while self.stream.connected:
            time_diff = time.time() - self.last_time  # in s
            time.sleep(0.005)
            if time_diff >= self.sampling_interval:
                self.last_time = time.time()

                logger.debug(f"Pull data - current time: {self.last_time}")
                logger.debug(f"time since last data pull {time_diff} seconds")

                if time_diff >= 2 * self.sampling_interval:
                    logger.warning(
                        "Feature computation time between two consecutive samples"
                        "was twice the feature sampling interval"
                    )
                if data is not None:
                    check_data = data

                data, timestamp = self.stream.get_data(winsize=self.winsize)
                if stream_start_time is None:
                    stream_start_time = timestamp[0]

                yield timestamp, data

                logger.info(f"Stream time: {timestamp[-1] - stream_start_time}")
