from datetime import datetime
import json
import multiprocessing
import multiprocessing.synchronize
import pathlib
import os
import queue

import numpy as np
import realtime_decoding
import pylsl
from numpy_ringbuffer import RingBuffer

import py_neuromodulation as nm


_Pathlike = str | os.PathLike


class Features(multiprocessing.Process):
    """Process class to calculate features from LSL stream."""

    def __init__(
        self,
        name: str,
        source_id: str,
        n_feats: int,
        sfreq: int,
        interval: float,
        queue_raw: multiprocessing.Queue,
        queue_features: multiprocessing.Queue,
        path_nm_channels: _Pathlike,
        path_nm_settings: _Pathlike,
        out_dir: _Pathlike,
        path_grids: str | None = None,
        line_noise: int | float | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(name=f"{name}Thread")
        self.interval = interval
        self.sfreq = sfreq
        self.queue_raw = queue_raw
        self.queue_features = queue_features
        self.path_nm_channels = pathlib.Path(path_nm_channels)
        self.path_nm_settings = pathlib.Path(path_nm_settings)
        self.out_dir = pathlib.Path(out_dir)
        self.finished = multiprocessing.Event()

        self.processor = nm.nm_run_analysis.DataProcessor(
            sfreq=self.sfreq,
            settings=path_nm_settings,
            nm_channels=path_nm_channels,
            line_noise=line_noise,
            path_grids=path_grids,
            verbose=verbose,
        )
        self.num_channels = len(self.processor.nm_channels)
        self.buffer = RingBuffer(
            capacity=self.sfreq,
            dtype=(float, self.num_channels),  # type: ignore
            allow_overwrite=True,
        )
        # Channels x Number of different features
        self.n_feats_total = (
            sum(self.processor.nm_channels["used"] == 1) * n_feats
        )
        self.source_id = source_id
        self.outlet = None
        self._save_settings()

    def _save_settings(self) -> None:
        # print("SAVING DATA ....")
        self.processor.nm_channels.to_csv(
            self.out_dir / self.path_nm_channels.name, index=False
        )
        with open(
            self.out_dir / self.path_nm_settings.name, "w", encoding="utf-8"
        ) as outfile:
            json.dump(self.processor.settings, outfile)
    
    def clear_queue(self) -> None:
        realtime_decoding.clear_queue(self.queue_raw)

    def run(self) -> None:
        info = pylsl.StreamInfo(
            name=self.name,
            type="EEG",
            channel_count=self.n_feats_total,
            nominal_srate=self.sfreq,
            channel_format="double64",
            source_id=self.source_id,
        )
        self.outlet = pylsl.StreamOutlet(info)

        while True:
            try:
                data = self.queue_raw.get(timeout=10.0)
            except queue.Empty:
                break
            else:
                # print("Got data")
                if data is None:
                    break

                # data = np.array(samples)  # shape (time, ch)
                self.buffer.extend(data.T)
                if not self.buffer.is_full:
                    continue
                features = self.processor.process(self.buffer[:].T)
                timestamp = np.datetime64(datetime.utcnow(), "ns")
                try:
                    self.queue_features.put(features, timeout=self.interval)
                except queue.Full:
                    print("Features queue Full. Skipping sample.")
                self.outlet.push_sample(
                    x=features.tolist(), timestamp=timestamp.astype(float)
                )
                # try:
                #     self.queue_features.get(block=False)
                # except queue.Empty:
                #     print("Features queue empty. Skipping sample.")
                #     continue

        try:
            self.queue_features.put(None, timeout=3.0)
        except queue.Full:
            pass
        self.clear_queue()
        print(f"Terminating: {self.name}")
