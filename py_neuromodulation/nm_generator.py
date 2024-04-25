import os
import time
import logging

logger = logging.getLogger("PynmLogger")

from typing import Iterator, Union
from pynput import keyboard
import numpy as np
import mne
from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL


class LSLOfflinePlayer:

    def __init__(
        self,
        settings: dict,
        stream_name: str = "example_stream",
        f_name: str = None,
        sfreq: Union[int, float] = None,
        data: np.array = None,
    ) -> None:

        self.settings = settings
        self.sfreq = sfreq
        self.stream_name = stream_name
        got_fname = f_name is not None
        got_sfreq_data = sfreq is not None and data is not None
        if not (got_fname or got_sfreq_data):
            error_msg = "Either f_name or sfreq and data must be provided."
            logger.critical(error_msg)
            raise ValueError(error_msg)

        if f_name is not None:
            self._path_raw = f_name

        if sfreq is not None and data is not None:

            info = mne.create_info(
                ch_names=[f"ch{i}" for i in range(data.shape[0])],
                ch_types=[f"dbs" for _ in range(data.shape[0])],
                sfreq=sfreq,
            )
            raw = mne.io.RawArray(data, info)
            self._path_raw = os.path.join(os.getcwd() + "temp_raw.fif")
            raw.save(self._path_raw, overwrite=True)

        self.player = PlayerLSL(
            self._path_raw, name = stream_name, chunk_size=100, n_repeat=1
        )
        self.interval = self.player.chunk_size / self.player.info["sfreq"]

        self.player = self.player.start()


class LSLStream:

    def __init__(
        self, settings: dict, stream_name: str
    ) -> None:

        self.stream_name = "example_stream"
        self.settings = settings
        try:
            self.stream = StreamLSL(name=stream_name, bufsize=2).connect(timeout=2)
        except Exception as e:
            msg =f"Could not connect to stream: {e}. No stream is running under the name {stream_name}"
            logger.warning(msg)
            raise RuntimeError(msg)
        # self.stream._inlet.recover = False
        self.winsize = (
            settings["segment_length_features_ms"] / self.stream.sinfo.sfreq
        )
        self.sampling_interval = 1 / self.settings["sampling_rate_features_hz"]

    def on_press(self, key):
        if key == keyboard.Key.esc:
            self.key_pressed = True
            return False

    def get_next_batch(self) -> np.array:
        self.last_time = time.time()
        check_data = None
        data = None
        
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        while self.stream.connected:
            time_diff = time.time() - self.last_time  # in s
            if time_diff >= self.sampling_interval:
                self.last_time = time.time()
                logger.info(f"Current time: {self.last_time}")

                if time_diff >= 2 * self.sampling_interval:
                    logger.warning(
                        "Feature computation time between two consecutive samples"
                        "was twice the feature sampling interval"
                    )
                if data is not None:
                    check_data = data

                data, timestamp = self.stream.get_data(winsize=self.winsize)
                # Checking if new data is incoming # TODO check for cleaner solution
                if data is not None and check_data is not None and np.array_equal(data, check_data):
                    logger.warning("No new data incoming. Stopping stream.")
                    self.stream.disconnect()

                yield timestamp, data
                if not listener.running:
                    logger.info("Keyboard interrupt")
                    self.stream.disconnect()


def raw_data_generator(
    data: np.ndarray,
    settings: dict,
    sfreq: int,
) -> Iterator[np.ndarray]:
    """
    This generator function mimics online data acquisition.
    The data are iteratively sampled with sfreq_new.
    Arguments
    ---------
        ieeg_raw (np array): shape (channels, time)
        sfreq: int
        sfreq_new: int
        offset_time: int | float
    Returns
    -------
        np.array: 1D array of time stamps
        np.array: new batch for run function of full segment length shape
    """
    sfreq_new = settings["sampling_rate_features_hz"]
    offset_time = settings["segment_length_features_ms"]
    offset_start = offset_time / 1000 * sfreq

    ratio_samples_features = sfreq / sfreq_new

    ratio_counter = 0
    for cnt in range(
        data.shape[1] + 1
    ):  # shape + 1 guarantees that the last sample is also included

        if (cnt - offset_start) >= ratio_samples_features * ratio_counter:

            ratio_counter += 1

            yield (
                np.arange(cnt - offset_start, cnt) / sfreq,  # what is this exactly? Not in main. need? 
                data[:, np.floor(cnt - offset_start).astype(int) : cnt]
            )
