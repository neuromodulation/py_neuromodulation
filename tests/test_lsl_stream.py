from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL
from mne_lsl import stream_viewer
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from py_neuromodulation import (nm_stream_abc, nm_plots, nm_IO, nm_define_nmchannels, nm_analysis, nm_stream_offline, nm_settings, nm_generator)
import mne
import threading
import time 
import random

(
    RUN_NAME,
    PATH_RUN,
    PATH_BIDS,
    PATH_OUT,
    datatype,
) = nm_IO.get_paths_example_data()

(
    raw,
    data,
    sfreq,
    line_noise,
    coord_list,
    coord_names,
) = nm_IO.read_BIDS_data(
    PATH_RUN=PATH_RUN, BIDS_PATH=PATH_BIDS, datatype=datatype
)

test_general_lsl = False
test_custom_lsl = True
f_name = "/Users/Sam/charite/py_neuro/py_neuromodulation/py_neuromodulation/data/sub-testsub/ses-EphysMedOff/ieeg/sub-testsub_ses-EphysMedOff_task-gripforce_run-0_ieeg.vhdr"


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
    settings = nm_settings.set_settings_fast_compute(settings)

    player = nm_generator.LSLOfflinePlayer(f_name = f_name, settings = settings)

    def get_example_stream(test_arr: np.array) -> nm_stream_abc.PNStream:

        settings["features"]["welch"] = True
        settings["features"]["fft"] = True
        settings["features"]["bursts"] = True
        settings["features"]["sharpwave_analysis"] = True
        settings["features"]["coherence"] = True
        settings["coherence"]["channels"] = [["LFP_RIGHT_0", "ECOG_RIGHT_0"]]
        settings["coherence"]["frequency_bands"] = ["high beta", "low gamma"]
        settings["sharpwave_analysis_settings"]["estimator"]["mean"] = []
        for sw_feature in list(settings["sharpwave_analysis_settings"]["sharpwave_features"].keys()):
            settings["sharpwave_analysis_settings"]["sharpwave_features"][sw_feature] = True
            settings["sharpwave_analysis_settings"]["estimator"]["mean"].append(sw_feature)


        nm_channels = nm_define_nmchannels.set_channels(
            ch_names=raw.ch_names,
            ch_types=raw.get_channel_types(),
            reference="default",
            bads=raw.info["bads"],
            new_names="default",
            used_types=("ecog", "dbs", "seeg"),
            target_keywords=["MOV_RIGHT"],
        )

        stream = nm_stream_offline.Stream(
            sfreq=1000, nm_channels=nm_channels, settings=settings, verbose=True
        )
        return stream


    arr = mne.io.read_raw_brainvision(f_name)
    stream = get_example_stream(arr)

    features = stream._run(arr, stream_lsl = True, plot_lsl = False)


    print("starting analysis")
    analyzer = nm_analysis.Feature_Reader(feature_dir = stream.PATH_OUT, feature_file = stream.PATH_OUT_folder_name)
    print(analyzer.feature_arr)



    analyzer.label_name = "MOV_RIGHT"
    analyzer.label = analyzer.feature_arr["MOV_RIGHT"]
    analyzer.feature_arr.iloc[100:108, -6:]
    print(analyzer._get_target_ch())



    analyzer.plot_target_averaged_channel(
    ch="ECOG_RIGHT_0",
    list_feature_keywords=None,
    epoch_len=4,
    threshold=0.5,
    ytick_labelsize=7,
    figsize_x=12,
    figsize_y=12,
    )
    plt.show()


    analyzer.plot_all_features(
    ytick_labelsize=6,
    clim_low=-2,
    clim_high=2,
    ch_used="ECOG_RIGHT_0",
    time_limit_low_s=0,
    time_limit_high_s=20,
    normalize=True,
    save=True,
    )
    plt.show()

    nm_plots.plot_corr_matrix(
    feature=analyzer.feature_arr.filter(regex="ECOG_RIGHT_0"),
    ch_name="ECOG_RIGHT_0-avgref",
    feature_names=analyzer.feature_arr.filter(
        regex="ECOG_RIGHT_0-avgref"
    ).columns,
    feature_file=analyzer.feature_file,
    show_plot=True,
    figsize=(15, 15),
    )
    plt.show()


    # features.label_name = "MOV_RIGHT"
    # features.label = features.feature_arr["MOV_RIGHT"]