from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL
import numpy as np
from py_neuromodulation import (nm_stream_abc, nm_IO, nm_define_nmchannels, nm_stream_offline, nm_settings, nm_generator)
import threading

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
f_name = f"{PATH_RUN}_ieeg.vhdr"
    
def test_general_lsl():
    if __name__ == "__main__":

        raw = nm_IO.read_raw_brainvision(f_name)

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

        def call_every_100ms():
            data, timestamps = stream1.get_data(winsize=100)
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


def test_offline_lsl():
    settings = nm_settings.get_default_settings()
    settings = nm_settings.set_settings_fast_compute(settings)

    player = nm_generator.LSLOfflinePlayer(f_name = f_name)

    def get_example_stream() -> nm_stream_abc.NMStream:

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

    stream = get_example_stream()

    features = stream.run(stream_lsl = True, plot_lsl= False, stream_lsl_name = "example_stream")

    assert features is not None
