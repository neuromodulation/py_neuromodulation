from collections.abc import Iterator
import time
from pynput import keyboard
import numpy as np
from py_neuromodulation import logger
from mne_lsl.lsl import resolve_streams


class LSLStream:
    """
    Class is used to create and connect to a LSL stream and pull data from it.
    """

    def __init__(self, settings: dict, stream_name: str | None = None) -> None:
        """
        Initialize the LSL stream.

        Parameters:
        -----------
        settings : dict
            Settings dictionary
        stream_name : str, optional
            Name of the stream to connect to. If not provided, the first available stream is used.

        Raises:
        -------
        RuntimeError
            If no stream is running under the provided name or if there are multiple streams running under the same name.
        """
        from mne_lsl.stream import StreamLSL

        self.settings = settings
        self._n_seconds_wait_before_disconnect = 3
        try:
            if stream_name is None:
                stream_name = resolve_streams()[0].name
                logger.info(f"Stream name not provided. Using first available stream: {stream_name}")
            self.stream = StreamLSL(name=stream_name, bufsize=2).connect(timeout=2)
        except Exception as e:
            msg = f"Could not connect to stream: {e}. Either no stream is running under the name {stream_name} or there is several streams under this name."
            logger.warning(msg)
            raise RuntimeError(msg)

        self.winsize = settings["segment_length_features_ms"] / self.stream.sinfo.sfreq
        self.sampling_interval = 1 / self.settings["sampling_rate_features_hz"]

    def on_press(self, key):
        """
        Function to stop the generator when the escape key is pressed.
        """
        if key == keyboard.Key.esc:
            self.key_pressed = True
            return False

    def get_next_batch(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        self.last_time = time.time()
        check_data = None
        data = None

        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        while self.stream.connected:
            time_diff = time.time() - self.last_time  # in s
            time.sleep(0.005)
            if time_diff >= self.sampling_interval:
                self.last_time = time.time()

                logger.info(f"Pull data - current time: {self.last_time}")

                if time_diff >= 2 * self.sampling_interval:
                    logger.warning(
                        "Feature computation time between two consecutive samples"
                        "was twice the feature sampling interval"
                    )
                if data is not None:
                    check_data = data

                data, timestamp = self.stream.get_data(winsize=self.winsize)

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

                if not listener.running:
                    logger.info("Keyboard interrupt")
                    self.stream.disconnect()
