from collections.abc import Iterator
import time
from typing import TYPE_CHECKING
import numpy as np
from py_neuromodulation.utils import logger
from mne_lsl.lsl import resolve_streams
import os

if TYPE_CHECKING:
    from py_neuromodulation import NMSettings


class LSLStream:
    """
    Class is used to create and connect to a LSL stream and pull data from it.
    """

    def __init__(self, settings: "NMSettings", stream_name: str | None = None) -> None:
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

        self.settings = settings
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

        self.winsize = settings.segment_length_features_ms / self.stream.sinfo.sfreq
        self.sampling_interval = 1 / self.settings.sampling_rate_features_hz

        # If not running the generator when the escape key is pressed.
        self.headless: bool = not os.environ.get("DISPLAY")
        # if not self.headless:
        # from py_neuromodulation.utils.keyboard import KeyboardListener

        # self.listener = KeyboardListener(("esc", self.set_keyboard_interrupt))
        # self.listener.start()

    def get_next_batch(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
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

                for i in range(self._n_seconds_wait_before_disconnect):
                    if (
                        data is not None
                        and check_data is not None
                        and np.allclose(data, check_data, atol=1e-7, rtol=1e-7)
                    ):
                        logger.warning(
                            f"No new data incoming. Disconnecting stream in {3-i} seconds."
                        )
                        time.sleep(1)
                        i += 1
                        if i == self._n_seconds_wait_before_disconnect:
                            self.stream.disconnect()
                            logger.warning("Stream disconnected.")
                            break

                yield timestamp, data

                logger.info(f"Stream time: {timestamp[-1] - stream_start_time}")

                # if not self.headless and self.keyboard_interrupt:
                #    logger.info("Keyboard interrupt")
                #    self.listener.stop()
                #    self.stream.disconnect()

    # def set_keyboard_interrupt(self):
    #    self.keyboard_interrupt = True
