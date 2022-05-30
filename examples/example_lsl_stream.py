from audioop import mul
import nm_stream_abc
import pandas as pd
import os
import multiprocessing
from threading import Timer
import numpy as np
import pylsl
import pickle
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from pynput.keyboard import Key, Listener
from py_neuromodulation import nm_lsl_stream, nm_define_nmchannels


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


def setup_stream():
    ch_names = [
        "FP1",
        "FP2",
        "C3",
        "C4",
        "P7",
        "P8",
        "O1",
        "O2",
    ]
    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=ch_names,
        ch_types=["eeg" for _ in range(len(ch_names))],
        reference=["average" for _ in range(len(ch_names))],
        bads=None,
        used_types=("eeg",),
    )

    stream = nm_lsl_stream.LSLStream(
        settings=None,
        nm_channels=nm_channels,
        path_grids=None,
        verbose=True,
    )

    for f in stream.settings["features"]:
        stream.settings["features"][f] = False
    stream.settings["features"]["fft"] = True

    for f in stream.settings["preprocessing"]:
        stream.settings["preprocessing"][f] = False
    stream.settings["preprocessing"]["re_referencing"] = True
    stream.settings["preprocessing"]["notch_filter"] = True
    stream.settings["preprocessing"]["preprocessing_order"] = [
        "re_referencing",
        "notch_filter",
    ]

    for f in stream.settings["postprocessing"]:
        stream.settings["postprocessing"][f] = False
    stream.settings["postprocessing"]["feature_normalization"] = True

    stream.settings["frequency_ranges_hz"] = {
        "theta": [4, 8],
        "alpha": [8, 12],
        "low beta": [13, 20],
        "high beta": [20, 35],
        "low gamma": [60, 80],
    }
    stream.init_stream(
        sfreq=250,
        line_noise=50,
        coord_list=None,
        coord_names=None,
    )

    # data = np.random.random([8, 250])

    return stream


class StreamApp:
    def __init__(
        self,
        VERBOSE: bool = False,
        TRAINING: bool = True,
        PREDICTION: bool = False,
        PATH_OUT: str = r"C:\Users\ICN_admin\Documents\LSL_Test",
        folder_name: str = "Test",
    ) -> None:

        self.stream = setup_stream()

        # try:
        #    self.stream.init_LSL()
        # except nm_lsl_stream.StreamNotFoundException as e:
        #    print(e)

        self.queue_raw = multiprocessing.Queue(1)
        self.queue_features = multiprocessing.Queue(1)
        self.VERBOSE = VERBOSE
        self.TRAINING = TRAINING
        self.PATH_OUT = PATH_OUT
        self.folder_name = folder_name

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

    def start_processes(
        self,
    ):

        processes = [
            multiprocessing.Process(
                target=self.get_features_wrapper,
                args=(
                    self.queue_raw,
                    self.queue_features,
                ),
            ),
            multiprocessing.Process(
                target=self.process_features,
                args=(self.queue_features,),
            ),
        ]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    def get_features_wrapper(
        self,
        queue_raw: multiprocessing.Queue,
        queue_features: multiprocessing.Queue,
    ):
        self.stream.init_stream(
            sfreq=250,
            line_noise=50,
            coord_list=None,
            coord_names=None,
        )  # somehow self.stream.fetures.features gets erased; re-init fixes that

        STOP_COND = False
        data_buffer = None
        while STOP_COND is False:
            data = queue_raw.get()

            if data is None:
                STOP_COND = True
                queue_features.put(None)
                break

            if data_buffer is None:
                data_buffer = data
            elif data_buffer.shape[1] < 250:
                data_buffer = np.concatenate((data_buffer, data), axis=1)
            elif data_buffer.shape[1] >= 250:
                # RingBuffer
                data_buffer = np.concatenate((data_buffer, data), axis=1)
                data_buffer = data_buffer[:, -250:]

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

    def process_features(self, queue_features: multiprocessing.Queue):
        # predict features based on model
        STOP_COND = False
        counter_samples = 0
        CREATE_FIG = True

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
                if counter_samples < 100:
                    print("right")
                    label = "right"
                elif counter_samples > 100 and counter_samples < 200:
                    print("left")
                    label = "left"
                elif counter_samples > 200 and counter_samples < 300:
                    print("rest")
                    label = "rest"
                elif counter_samples > 300:
                    print("press esc training done")
                    label = "rest"
                features_comp["label"] = label
            if self.PREDICTION is True:
                # predict = self.model.predict(np.expand_dims(features_comp, axis=0))
                prediction_proba = self.model.predict_proba(
                    np.expand_dims(features_comp, axis=0)
                )[0, :]

                def update_plt(axes):
                    axes.bar([0, 1, 2], prediction_proba, color="blue")
                    # axes.figure.canvas.draw()

                def update(
                    frame, prediction_proba
                ):  # here frame needs to be accepted by the function since this is used in FuncAnimations
                    # ax.bar([0, 1, 2], prediction_proba, color="blue")
                    ln.set_data([0, 1, 2], prediction_proba)
                    return (ln,)

                if CREATE_FIG is True:
                    CREATE_FIG = False
                    
                    # timer = fig.canvas.new_timer(interval=100)
                    # timer.add_callback(update_plt, ax)
                    # timer.start()

                # ax.bar([0, 1, 2], prediction_proba, color="blue")
                print(self.model.predict(np.expand_dims(features_comp, axis=0)))
                # plt.show()

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


if __name__ == "__main__":

    app = StreamApp(VERBOSE=False, TRAINING=False, PREDICTION=True)

    streams = pylsl.resolve_streams(
        wait_time=1,
    )
    lsl_streaminlet = pylsl.StreamInlet(info=streams[0], max_buflen=25)

    def on_press(key):
        print("{0} pressed".format(key))

    def on_release(key):
        if key == Key.esc:
            # Stop listener
            print("reiceived stop key pressed")

            app.queue_raw.put(None)
            timer.cancel()
            return False

    time_call_get_data_s = np.round(
        1 / app.stream.settings["sampling_rate_features_hz"], 2
    )

    def get_data(queue_raw: multiprocessing.Queue):
        samples, _ = lsl_streaminlet.pull_chunk(max_samples=25, timeout=1)
        raw_data = np.array(samples).T  # shape (ch, time)
        queue_raw.put(raw_data)

    timer = RepeatTimer(time_call_get_data_s, get_data, args=(app.queue_raw,))
    timer.start()

    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()

    plt.show()

    app.start_processes()
