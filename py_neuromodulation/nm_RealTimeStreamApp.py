from multiprocessing import managers
import os
import multiprocessing
from multiprocessing.managers import BaseManager
from queue import LifoQueue
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from py_neuromodulation import nm_stream_abc


class MyManager(BaseManager):
    pass


MyManager.register("LifoQueue", LifoQueue)


class StreamApp:
    def __init__(
        self,
        stream: nm_stream_abc.PNStream,
        VERBOSE: bool = False,
        TRAINING: bool = False,
        PREDICTION: bool = True,
        PATH_OUT: str = r"C:\Users\ICN_admin\Documents\LSL_Test",
        folder_name: str = "Test",
        training_samples_each_cond_s: int = 10,
    ) -> None:

        self.stream = stream
        self.queue_raw = multiprocessing.Queue(1)
        self.queue_features = multiprocessing.Queue(1)
        manager = MyManager()
        manager.start()
        self.queue_plotting = manager.LifoQueue()
        self.VERBOSE = VERBOSE
        self.TRAINING = TRAINING
        self.PATH_OUT = PATH_OUT
        self.folder_name = folder_name
        self.samples_buffer_batch = int(
            self.stream.sfreq
            * self.stream.settings["segment_length_features_ms"]
            / 1000
        )
        self.training_samples = (
            training_samples_each_cond_s
            * self.stream.settings["sampling_rate_features_hz"]
        )

        self.PREDICTION = PREDICTION
        if PREDICTION is True:
            with open(
                os.path.join(
                    self.PATH_OUT, self.folder_name, "linear_model.pkl"
                ),
                "rb",
            ) as fid:
                self.model = pickle.load(fid)
        # Test init Stream:

        # Test get data:
        # dat, _ = lsl_streaminlet.pull_chunk()

        # dat, _ = self.stream.lsl_streaminlet.pull_chunk(
        #    max_samples=250, timeout=5
        # )
        # dat = np.array(dat).T
        # f = self.stream.run_analysis.process_data(dat)

    def get_features_wrapper(
        self,
        queue_raw: multiprocessing.Queue,
        queue_features: multiprocessing.Queue,
    ):
        # BUG: self.stream.fetures.features gets erased; this re-init fixes that
        self.stream.init_stream(
            sfreq=250,
            line_noise=50,
            coord_list=None,
            coord_names=None,
        )

        STOP_COND = False
        data_buffer = None
        while STOP_COND is False:
            data = queue_raw.get()

            if data is None:
                STOP_COND = True
                queue_features.put(None)
                break

            # buffering data enables settings["segment_length_features_ms"]
            # data is streamed with every 1 / settings["sampling_rate_features_hz"]
            # the data streamed into queue_raw is always new data
            # therefore the queue should contain settings["sampling_rate_features_hz"]
            # all novel data
            # in the case of OpenBCI pylsl the sfreq = 250, for
            # settings["sampling_rate_features_hz"] = 10 Hz; 25 samples should therefore
            # every 100 ms be sampled

            if data_buffer is None:
                data_buffer = data
            elif data_buffer.shape[1] < self.samples_buffer_batch:
                data_buffer = np.concatenate((data_buffer, data), axis=1)
            elif data_buffer.shape[1] >= self.samples_buffer_batch:
                # RingBuffer
                data_buffer = np.concatenate((data_buffer, data), axis=1)
                data_buffer = data_buffer[:, -self.samples_buffer_batch :]

                if self.VERBOSE is True:
                    print(
                        f"get features_wrapper: data.shape: {data_buffer.shape}"
                    )
                # enough samples to calc features
                features_comp = self.stream.run(data_buffer)
                if self.VERBOSE is True:
                    print(
                        f"estimated features with shape {features_comp.shape}"
                    )
                queue_features.put(features_comp)

    def process_features(
        self,
        queue_features: multiprocessing.Queue,
        queue_plotting: multiprocessing.Queue,
    ):

        STOP_COND = False
        counter_samples = 0
        CREATE_FIG = False

        feature_df = []

        while STOP_COND is False:
            features_comp = queue_features.get()

            if features_comp is None:
                STOP_COND = True
                if self.VERBOSE is True:
                    print("stop session")
                self.terminate_session(feature_df)
                break
            if self.TRAINING is True:
                counter_samples += 1
                if counter_samples < self.training_samples:
                    print("right")
                    label = "right"
                elif (
                    counter_samples > self.training_samples
                    and counter_samples < (2 * self.training_samples)
                ):
                    print("left")
                    label = "left"
                elif (
                    counter_samples > 2 * self.training_samples
                    and counter_samples < 3 * self.training_samples
                ):
                    print("rest")
                    label = "rest"
                elif counter_samples > 3 * self.training_samples:
                    print("press esc training done")
                    label = "rest"
                features_comp["label"] = label

            if self.PREDICTION is True:
                # predict = self.model.predict(np.expand_dims(features_comp, axis=0))
                prediction_proba = self.model.predict_proba(
                    np.expand_dims(features_comp, axis=0)
                )[0, :]
                queue_plotting.put(prediction_proba)

                print(self.model.predict(np.expand_dims(features_comp, axis=0)))

            feature_df.append(features_comp)

            if self.VERBOSE is True:
                print(
                    f"received features in process_features with shape {features_comp.shape}"
                )

    def terminate_session(self, feature_df: list):
        if self.VERBOSE is True:
            print("save features")
        if self.TRAINING is True and self.PREDICTION is False:
            self.stream.save_after_stream(
                self.PATH_OUT,
                self.folder_name,
                feature_arr=pd.DataFrame(feature_df),
            )
