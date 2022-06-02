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
from py_neuromodulation import (
    nm_lsl_stream,
    nm_define_nmchannels,
    nm_RealTimeStreamApp,
)


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

    return stream


def main():

    stream_ = setup_stream()

    app = nm_RealTimeStreamApp.StreamApp(
        stream_,
        VERBOSE=False,
        TRAINING=False,
        PREDICTION=True,
        fig=None,
        ax=None,
    )

    # pylsl needs to be started from the main thread
    # therefore the get_data function cannot be set in nm_stream
    streams = pylsl.resolve_streams(
        wait_time=1,
    )

    lsl_streaminlet = pylsl.StreamInlet(info=streams[0], max_buflen=25)

    def get_data(queue_raw: multiprocessing.Queue):
        samples, _ = lsl_streaminlet.pull_chunk(max_samples=25, timeout=1)
        raw_data = np.array(samples).T  # shape (ch, time)
        queue_raw.put(raw_data)

    # the KeyListener is also a Thread object
    def on_press(key):
        print("{0} pressed".format(key))

    def on_release(key):
        if key == Key.esc:
            # Stop listener
            print("reiceived stop key pressed")

            app.queue_raw.put(None)
            timer.cancel()
            return False

    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()

    def update(
        frame, queue_plotting: multiprocessing.Queue
    ):  # here frame needs to be accepted by the function since this is used in FuncAnimations
        data = queue_plotting.get()  # this blocks untill it gets some data
        ln.set_data([0, 1, 2], data)
        return (ln,)

    fig, ax = plt.subplots()
    (ln,) = plt.plot([], [], "ro")
    ani = FuncAnimation(fig, update, blit=True, fargs=(app.queue_plotting,))
    plt.show()

    time_call_get_data_s = np.round(
        1 / app.stream.settings["sampling_rate_features_hz"], 2
    )

    # RepeatTimer will allow synchronized data pull
    timer = RepeatTimer(time_call_get_data_s, get_data, args=(app.queue_raw,))
    timer.start()

    app.start_processes()


if __name__ == "__main__":

    main()
