from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL
from mne_lsl import stream_viewer
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from py_neuromodulation import (nm_stream_abc, nm_define_nmchannels, nm_analysis, nm_stream_offline, nm_settings, nm_generator)
import mne
import threading
import random


test_general_lsl = True
test_custom_lsl = True
f_name = "./py_neuromodulation/data/sub-testsub/ses-EphysMedOff/ieeg/sub-testsub_ses-EphysMedOff_task-gripforce_run-0_ieeg.vhdr"


if test_general_lsl:
    print("General LSL test started")
    if __name__ == "__main__":

        raw = mne.io.read_raw_brainvision(f_name)
        # plt.figure()
        # plt.plot([1, 4, 5])
        # plt.show(block=True)

        player = PlayerLSL(f_name, name="example_stream", chunk_size=100)
        player = player.start()

        player.info

        sfreq = player.info["sfreq"]

        chunk_size = player.chunk_size
        interval = chunk_size / sfreq  # in seconds
        print(f"Interval between 2 push operations: {interval} seconds.")

        stream = StreamLSL(name="example_stream", bufsize=2).connect()
        ch_types = stream.get_channel_types(unique=True)
        print(f"Channel types included: {', '.join(ch_types)}")

        # viewer = stream_viewer.StreamViewer(stream_name="example_stream")
        # viewer.start()

        data_l = []
        timestamps_l = []
        idx_ = 0

        # start = time.time()
        def call_every_100ms():
            data, timestamps = stream.get_data(winsize=100)
            print(data.shape)
            data_l.append(data)
            timestamps_l.append(timestamps)

        t = threading.Timer(0.1, call_every_100ms)
        t.start()

        import time

        time_start = time.time()

        while time.time() - time_start <= 10:
            time.sleep(1)
        t.cancel()

        #    while idx_ < 100:
        #        if stream.n_new_samples >= 100:

        # now = time.time()
        # if now - start >= 0.1:
        #    #do_stuff()
        #    start = now

        # time.sleep(1)

        # plt.figure()
        # plt.plot(timestamps)
        # plt.show(block=True)
        # check here 1. the timing
        print(np.concatenate(data_l).shape)

        stream.disconnect()
        player.stop()



if test_custom_lsl:
    print("Custom LSL test started")
    # settings 
    settings = nm_settings.get_default_settings()
    settings = nm_settings.reset_settings(settings)
    settings['features']['fft'] = True
    settings['features']['bursts'] = True
    settings['features']['sharpwave_analysis'] = True

    # player (mock lsl stream)
    player = nm_generator.LSLOfflinePlayer(f_name = f_name, settings = settings)

    def get_example_stream(test_arr: np.array) -> nm_stream_abc.PNStream:
            settings = nm_settings.get_default_settings()
            settings["features"]["raw_hjorth"] = True
            settings["features"]["bandpass_filter"] = True
            settings["features"]["stft"] = True
            settings["features"]["fft"] = True
            settings["features"]["sharpwave_analysis"] = True
            settings["features"]["fooof"] = True
            settings["features"]["bursts"] = True
            settings["features"]["linelength"] = True
            settings["features"]["nolds"] = False
            settings["features"]["mne_connectivity"] = False
            settings["features"]["return_raw"] = True
            settings["features"]["coherence"] = False

            nm_channels = nm_define_nmchannels.get_default_channels_from_data(test_arr)

            stream = nm_stream_offline.Stream(
                sfreq=1000, nm_channels=nm_channels, settings=settings, verbose=True
            )
            return stream


    arr = mne.io.read_raw_brainvision(f_name).get_data()
    stream = get_example_stream(arr) 
    # viewer = stream_viewer.StreamViewer(stream_name="example_stream")
    stream.run(arr, stream_lsl = True)
    # df = stream.run(arr, stream_lsl = True, stream_lsl_name = "example_stream")
    # df.head()
    print("starting analysis")
    analyzer = nm_analysis.Feature_Reader(feature_dir = stream.PATH_OUT, feature_file = stream.PATH_OUT_folder_name)
    print(analyzer.feature_arr)