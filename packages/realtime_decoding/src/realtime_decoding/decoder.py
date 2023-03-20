import multiprocessing
import multiprocessing.synchronize
import pathlib
import pickle
import queue
import tkinter
import tkinter.filedialog
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pylsl

import realtime_decoding

from .helpers import _PathLike

_timezone = timezone.utc


class Decoder(multiprocessing.Process):
    """Make predictions in real time."""

    def __init__(
        self,
        queue_decoding: multiprocessing.Queue,
        queue_features: multiprocessing.Queue,
        interval: float,
        out_dir: _PathLike,
        verbose: bool,
        model_path: _PathLike,
    ) -> None:
        super().__init__(name="DecodingProcess")
        self.queue_decoding = queue_decoding
        self.queue_feat = queue_features
        self.interval = interval
        self.verbose = verbose
        self.out_dir = pathlib.Path(out_dir)

        self._threshold: float = 0.5

        self.filename = pathlib.Path(model_path)

        with open(self.filename, "rb") as file:
            self._model = pickle.load(file)
        self._save_model()

    def _save_model(self) -> None:
        with open(self.out_dir / self.filename.name, "wb") as file:
            pickle.dump(self._model, file)

    def clear_queue(self) -> None:
        for q in (self.queue_feat, self.queue_decoding):
            realtime_decoding.clear_queue(q)

    def run(self) -> None:
        labels = ["Prediction", "Probability"]  # "Threshold"

        info = pylsl.StreamInfo(
            name="Decoding",
            type="EEG",
            channel_count=2,
            channel_format="double64",
            source_id="decoding_1",
        )
        channels = info.desc().append_child("channels")
        for label in labels:
            channels.append_child("channel").append_child_value("label", label)
        outlet = pylsl.StreamOutlet(info)
        while True:
            try:
                sample = self.queue_feat.get(timeout=10.0)
            except queue.Empty:
                break
            else:
                if self.verbose:
                    print("Got features.")
                if sample is None:
                    print("Found None value, terminating decoder process.")
                    break

                # Predict
                sample_ = sample[[i for i in sample.index if i != "label_train"]]

                dat_pr = np.nan_to_num(np.expand_dims(sample_.to_numpy(), 0))
                y = float(self._model.predict_proba(dat_pr)[0, 1])
                print(f"pr: {y}")

                timestamp = np.datetime64(datetime.now(_timezone), "ns")

                output = pd.DataFrame(
                    [[y >= self._threshold, y]],  # self._threshold
                    columns=labels,
                    index=[timestamp],
                )
                outlet.push_sample(
                    x=list(output.to_numpy().squeeze()),
                    timestamp=timestamp.astype(float),
                )
        try:
            self.queue_decoding.put(None, timeout=3.0)
        except queue.Full:
            pass
        self.clear_queue()
        print(f"Terminating: {self.name}")