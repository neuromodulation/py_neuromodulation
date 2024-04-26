from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL
import numpy as np
from py_neuromodulation import (nm_stream_abc, nm_IO, nm_define_nmchannels, nm_analysis, nm_stream_offline, nm_settings, nm_generator)
import mne
import threading
import time 

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


test_general_lsl = True
test_offline_lsl = True
test_live_lsl = False
f_name = f"{PATH_RUN}_ieeg.vhdr"

    
if test_general_lsl:
    print("General LSL test started")
    if __name__ == "__main__":

        raw = mne.io.read_raw_brainvision(f_name)

        player1 = PlayerLSL(f_name, name="general_stream", chunk_size=100)
        player1 = player1.start()

        player1.info

        sfreq = player1.info["sfreq"]

        chunk_size = player1.chunk_size
        interval = chunk_size / sfreq  # in seconds
        print(f"Interval between 2 push operations: {interval} seconds.")

        stream1 = StreamLSL(name="general_stream", bufsize=2).connect()
        ch_types = stream1.get_channel_types(unique=True)
        print(f"Channel types included: {', '.join(ch_types)}")

        data_l = []
        timestamps_l = []
        idx_ = 0

        # start = time.time()
        def call_every_100ms():
            data, timestamps = stream1.get_data(winsize=100)
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

        print(np.concatenate(data_l).shape)

        stream1.disconnect()
        player1.stop()
        time.sleep(5)


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


if test_offline_lsl:
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

    features = stream._run(arr, stream_lsl = True, stream_lsl_name = "example_stream")
    # features = stream._run(arr, stream_lsl = True, plot_lsl = False, )


    print("starting analysis")
    analyzer = nm_analysis.Feature_Reader(feature_dir = stream.PATH_OUT, feature_file = stream.PATH_OUT_folder_name)
    print(analyzer.feature_arr)



    analyzer.label_name = "MOV_RIGHT"
    analyzer.label = analyzer.feature_arr["MOV_RIGHT"]
    analyzer.feature_arr.iloc[100:108, -6:]
    print(analyzer._get_target_ch())


if test_live_lsl:

    settings = nm_settings.get_default_settings()
    settings = nm_settings.set_settings_fast_compute(settings)

    def get_example_stream(test_arr: np.array) -> nm_stream_abc.PNStream:

        settings["features"]["welch"] = False
        settings["features"]["fft"] = True
        settings["features"]["bursts"] = True
        settings["features"]["sharpwave_analysis"] = True
        settings["features"]["coherence"] = True
        settings["coherence"]["channels"] = [["0", "1", "2", "3", "4", "5", "6", "7"]]
        settings["coherence"]["frequency_bands"] = ["high beta", "low gamma"]
        settings["sharpwave_analysis_settings"]["estimator"]["mean"] = []
        for sw_feature in list(settings["sharpwave_analysis_settings"]["sharpwave_features"].keys()):
            settings["sharpwave_analysis_settings"]["sharpwave_features"][sw_feature] = True
            settings["sharpwave_analysis_settings"]["estimator"]["mean"].append(sw_feature)


        ch_names = [f"{i}" for i in range(8)]
        ch_types=['EEG']*8
        bads=raw.info["bads"]

        nm_channels = nm_define_nmchannels.set_channels(ch_names=ch_names, ch_types=ch_types, bads = bads, new_names="default", used_types=("ecog", "dbs", "eeg", "eeg"), target_keywords=["MOV_RIGHT"])

        stream = nm_stream_offline.Stream(
            sfreq=250, nm_channels=nm_channels, settings=settings, verbose=True
        )
        return stream
    

    arr = mne.io.read_raw_brainvision(f_name)
    stream = get_example_stream(arr)

    features = stream._run(arr, stream_lsl = True, plot_lsl = False, stream_lsl_name = "obci_eeg1")