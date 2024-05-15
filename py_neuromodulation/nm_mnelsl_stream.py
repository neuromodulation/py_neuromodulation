from collections.abc import Iterator
from pathlib import Path
import time
from pynput import keyboard
import numpy as np
from mne_lsl.stream import StreamLSL
from mne_lsl.lsl import resolve_streams
from py_neuromodulation import logger


class LSLStream:

    def __init__(self, settings: dict, stream_name: str = None) -> None:

        self.settings = settings
        try:
            if stream_name is None:
                stream_name = resolve_streams()[0].name
                logger.info(f"Stream name not provided. Using first available stream: {stream_name}")
            self.stream = StreamLSL(name=stream_name, bufsize=2).connect(timeout=2)
        except Exception as e:
            msg = f"Could not connect to stream: {e}. No stream is running under the name {stream_name}"
            logger.warning(msg)
            raise RuntimeError(msg)

        self.winsize = settings["segment_length_features_ms"] / self.stream.sinfo.sfreq
        self.sampling_interval = 1 / self.settings["sampling_rate_features_hz"]

    def on_press(self, key):
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

                for i in range(3):
                    if (
                        data is not None
                        and check_data is not None
                        and np.array_equal(data, check_data)
                    ):
                        logger.warning(
                            f"No new data incoming. Disconnecting stream in {3-i} seconds."
                        )
                        time.sleep(1)
                        i += 1
                        if i == 3:
                            self.stream.disconnect()
                            logger.warning("Stream disconnected.")
                            break

                yield timestamp, data

                if not listener.running:
                    logger.info("Keyboard interrupt")
                    self.stream.disconnect()
