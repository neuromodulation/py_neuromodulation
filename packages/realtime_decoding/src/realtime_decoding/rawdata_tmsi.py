import math
import queue
import time
import multiprocessing
import threading

import numpy as np
import realtime_decoding
from py_neuromodulation import nm_filter


class RawDataTMSi(multiprocessing.Process):  # threading.Thread):
    def __init__(
        self,
        interval: float,
        sfreq: int | float,
        num_channels: int,
        queue_source: queue.Queue,
        queue_raw: queue.Queue,
        queue_emg: queue.Queue,
    ) -> None:
        super().__init__(name="RawTMSiThread", daemon=True)
        print(f"Initializing RawTMSiThread... ")
        self.sfreq = sfreq
        self.num_channels = num_channels

        self.queue_source = queue_source
        self.queue_raw = queue_raw
        self.queue_emg = queue_emg

        # Output data every *interval* seconds
        self.interval = interval
        self.capacity = math.floor(self.sfreq * self.interval)
        self.buffer = np.empty(shape=(self.num_channels, 0))

        self.notch_filter = nm_filter.IIRFilter(
            sfreq=self.sfreq,
            order=2,
            l_freq=48,
            h_freq=52,
            filter_type="bandstop",
        )
        self.emg_filter = nm_filter.IIRFilter(
            sfreq=self.sfreq,
            order=2,
            l_freq=15,
            h_freq=500,
            filter_type="bandpass",
        )

    def clear_queue(self) -> None:
        realtime_decoding.clear_queue(self.queue_source)

    def run(self) -> None:
        """Method that retrieves samples from the queue, reshapes them into
        the desired format and filters the samples.
        """

        def put(
            raw_data: np.ndarray | None,
            emg: np.ndarray | None,
            interval: float,
        ) -> None:
            try:
                self.queue_raw.put(raw_data, timeout=interval)
            except queue.Full:
                print("Raw out queue Full. Skipping sample.")
            try:
                self.queue_emg.put(emg, timeout=interval)
            except queue.Full:
                print("EMG out queue Full. Skipping sample.")

        start = time.time()
        while True:
            try:
                sd = self.queue_source.get(timeout=10.0)
            except queue.Empty:
                break
            else:
                # self.queue_in.task_done()
                print("Found raw input sample.")

                # Reshape the samples retrieved from the queue
                samples = np.reshape(
                    sd.samples,
                    (sd.num_samples_per_sample_set, sd.num_sample_sets),
                    order="F",
                )
                # print(f"{samples.shape = }")

                self.buffer = np.concatenate(
                    (self.buffer, samples[:, :]), axis=1
                )
                if self.buffer.shape[1] >= self.capacity:
                    data = self.notch_filter.process(self.buffer)
                    emg = self.emg_filter.process(data.copy())
                    put(raw_data=data, emg=emg, interval=self.interval)
                    self.buffer = np.empty(shape=(self.num_channels, 0))
                    # print(f"Time elapsed: {time.time()-start :.4f} sec")
                    start = time.time()

        put(raw_data=None, emg=None, interval=3.0)
        self.clear_queue()
        print(f"Terminating: {self.name}")
