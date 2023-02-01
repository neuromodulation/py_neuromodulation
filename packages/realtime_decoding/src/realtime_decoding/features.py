import json
import multiprocessing
import multiprocessing.synchronize
import pathlib
import queue
import tkinter
import tkinter.filedialog
from datetime import datetime

import numpy as np
import py_neuromodulation as nm
import pylsl
from numpy_ringbuffer import RingBuffer

import realtime_decoding

from .helpers import _PathLike


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
        out_dir: _PathLike,
        verbose: bool,
        path_grids: str | None = None,
        line_noise: int | float | None = None,
        training_samples: int = 60,
        training_enabled: bool = False,
    ) -> None:
        super().__init__(name=f"{name}Process")
        self.interval = interval
        self.sfreq = sfreq
        self.queue_raw = queue_raw
        self.queue_features = queue_features
        self.verbose = verbose
        self.out_dir = pathlib.Path(out_dir)
        self.finished = multiprocessing.Event()

        self.paths = {}
        for keyword, ftype in (
            ("nm_channels", "csv"),
            ("nm_settings", "json"),
        ):
            filename = tkinter.filedialog.askopenfilename(
                title=f"Select {keyword} file",
                filetypes=(("Files", f"*.{ftype}*"),),
            )
            self.paths[keyword] = pathlib.Path(filename)

        self.processor = nm.nm_run_analysis.DataProcessor(
            sfreq=self.sfreq,
            settings=self.paths["nm_settings"],
            nm_channels=self.paths["nm_channels"],
            line_noise=line_noise,
            path_grids=path_grids,
            verbose=self.verbose,
        )
        self.num_channels = len(self.processor.nm_channels)
        self.buffer = RingBuffer(
            capacity=self.sfreq,
            dtype=(float, self.num_channels),  # type: ignore
            allow_overwrite=True,
        )
        # Channels * Number of different features
        self.n_feats_total = (
            sum(self.processor.nm_channels["used"] == 1) * n_feats
        )
        self.source_id = source_id
        self.outlet = None
        self._save_settings()

        print(f"value of training enabled: {training_enabled}")
        self.training_enabled = training_enabled
        if training_enabled is True:
            print("training is enabled")
            self.training_counter = 0
            self.training_samples = training_samples
            self.training_class = 0 # REST

            # the labels are sent as an additional LSL channel
            self.n_feats_total = self.n_feats_total + 1

    def _save_settings(self) -> None:
        # print("SAVING DATA ....")
        self.processor.nm_channels.to_csv(
            self.out_dir / self.paths["nm_channels"].name, index=False
        )
        with open(
            self.out_dir / self.paths["nm_settings"].name,
            "w",
            encoding="utf-8",
        ) as outfile:
            json.dump(self.processor.settings, outfile)

    def clear_queue(self) -> None:
        realtime_decoding.clear_queue(self.queue_raw)

    def run(self) -> None:
        while True:
            try:
                sd = self.queue_raw.get(timeout=10.0)
                # data = self.queue_raw.get(timeout=10.0)
            except queue.Empty:
                break
            else:
                # print("Got data")
                if sd is None:
                    print("Found None value, terminating features process.")
                    break
                if self.verbose:
                    print("Found raw input sample.")
                # Reshape the samples retrieved from the queue
                data = np.reshape(
                    sd.samples,
                    (sd.num_samples_per_sample_set, sd.num_sample_sets),
                    order="F",
                )
                # data = np.array(samples)  # shape (time, ch)
                self.buffer.extend(data.T)
                if not self.buffer.is_full:
                    continue
                features = self.processor.process(self.buffer[:].T)
                timestamp = np.datetime64(datetime.utcnow(), "ns")

                if self.training_enabled is True:

                    # the analog channel data is stored in self.buffer
                    # this channel can be added to the calculated features, and simply finished with escape
                    #print(self.buffer[:].T)
                    #print(f"buffer shape: {self.buffer.shape}")
                    features["label_train"] = np.mean(self.buffer[-409:, 24])  # get index from analog 
                try:
                    self.queue_features.put(features, timeout=self.interval)
                except queue.Full:
                    if self.verbose:
                        print("Features queue Full. Skipping sample.")
                if self.outlet is None:
                    info = pylsl.StreamInfo(
                        name=self.name,
                        type="EEG",
                        channel_count=self.n_feats_total,
                        nominal_srate=self.sfreq,
                        channel_format="double64",
                        source_id=self.source_id,
                    )
                    channels = info.desc().append_child("channels")
                    for label in features.index:
                        channels.append_child("channel").append_child_value(
                            "label", label
                        )
                    self.outlet = pylsl.StreamOutlet(info)
                self.outlet.push_sample(
                    x=features.tolist(), timestamp=timestamp.astype(float)
                )

        try:
            self.queue_features.put(None, timeout=3.0)
        except queue.Full:
            pass
        self.clear_queue()
        print(f"Terminating: {self.name}")
