from datetime import datetime, timezone
import multiprocessing
import multiprocessing.synchronize
import os
import pathlib
import pickle
import queue
import tkinter
import tkinter.filedialog

import numpy as np
import pandas as pd
import realtime_decoding
import pylsl
import sklearn.dummy
import sklearn.model_selection




_timezone = timezone.utc


_Pathlike = str | os.PathLike


class IntentionDecoder(multiprocessing.Process):
    """Decode motor intention in real time."""

    def __init__(
        self,
        queue_decoding: multiprocessing.Queue,
        queue_features: multiprocessing.Queue,
        interval: float,
        out_dir: _Pathlike,
    ) -> None:
        super().__init__(name="DecodingThread")
        self.queue_decoding = queue_decoding
        self.queue_feat = queue_features
        self.interval = interval
        self.out_dir = pathlib.Path(out_dir)

        self._threshold: float = 0.5

        self._prediction_buffer_len: pd.Timedelta = pd.Timedelta(seconds=1.0)
        self._prediction_buffer: pd.DataFrame | None = None

        root = tkinter.Tk()
        filename = tkinter.filedialog.askopenfilename(
            title = 'Select model', 
            filetypes = (
                ('pickle files',[ '*.p', "*.pkl", "*.pickle"]),
                ('All files', '*.*'),
            )
        )
        root.withdraw()
        self.filename = pathlib.Path(filename)
        
        # self._model = sklearn.dummy.DummyClassifier(strategy="stratified")
        with open(self.filename, "rb") as file:
            self._model= pickle.load(file)
        self._save_model()


    def _save_model(self) -> None:
        with open(
            self.out_dir / self.filename.name, "wb"
        ) as file:
            pickle.dump(self._model, file)

        
    def clear_queue(self) -> None:
        for q in (self.queue_feat, self.queue_decoding):
            realtime_decoding.clear_queue(q)

    def run(self) -> None:
        def _predict(data) -> None:
            y = self._model.predict(data)

            timestamp = np.datetime64(datetime.now(_timezone), "ns")
            output = pd.DataFrame(
                [[y >= self._threshold, y, self._threshold]],
                columns=["Prediction", "Probability", "Threshold"],
                index=[timestamp],
            )

            try:
                self.queue_decoding.put(output, timeout=self.interval)
            except queue.Full:
                print("Features queue Full. Skipping sample.")
            self.outlet.push_sample(
                x=output.tolist(), timestamp=timestamp.astype(float)
            )
            try:
                self.queue_decoding.get(block=False)
            except queue.Empty:
                print("Features queue empty. Skipping sample.")


        info = pylsl.StreamInfo(
            name="Decoding",
            type="EEG",
            channel_count=3,
            channel_format="double64",
            source_id="decoding_1",
        )
        self.outlet = pylsl.StreamOutlet(info)
        while True:
            try:
                sample = self.queue_feat.get(timeout=10.0)
            except queue.Empty:
                break
            else:
                print("Got features.")
                if sample is None:
                    break
                _predict(sample)
        try:
            self.queue_decoding.put(None, timeout=3.0)
        except queue.Full:
            pass
        self.clear_queue()
        print(f"Terminating: {self.name}")
