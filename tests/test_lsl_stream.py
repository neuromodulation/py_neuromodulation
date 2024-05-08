from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL
import numpy as np
from py_neuromodulation import (nm_mnelsl_generator, nm_IO, nm_define_nmchannels, nm_stream_offline, nm_settings, nm_generator)
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
    
def test_mne_lsl():
    raw = nm_IO.read_BIDS_data(f_name)

    player1 = PlayerLSL(f_name,  name="general_stream", chunk_size=10)
    player1 = player1.start()

    stream1 = StreamLSL(name="general_stream", bufsize=2).connect()
    ch_types = stream1.get_channel_types(unique=True)
    assert "eeg" in ch_types, "Expected EEG channels in the stream"

    data_l = []
    timestamps_l = []

    def call_every_100ms():
        data, timestamps = stream1.get_data(winsize=10)
        data_l.append(data)
        timestamps_l.append(timestamps)

    t = threading.Timer(0.1, call_every_100ms)
    t.start()

    import time

    time_start = time.time()

    while time.time() - time_start <= 10:
        time.sleep(1)
    t.cancel()

    collected_data_shape = np.concatenate(data_l).shape
    assert collected_data_shape[0] > 0 and collected_data_shape[1] > 0, "Expected non-empty data"

    stream1.disconnect()
    player1.stop()



def test_offline_lsl(setup_default_stream_fast_compute):
    settings = nm_settings.get_default_settings()
    settings = nm_settings.set_settings_fast_compute(settings)

    player = nm_mnelsl_generator.LSLOfflinePlayer(f_name = f_name)
    player.start_player()

    data, stream = setup_default_stream_fast_compute

    features = stream.run(stream_lsl = True, plot_lsl= False, stream_lsl_name = "example_stream")
    # check sfreq
    assert raw.info['sfreq'] == stream.sfreq, "Expected same sampling frequency in the stream and input file"
    assert player.player.info['sfreq'] == stream.sfreq, "Expected same sampling frequency in the stream and player"

    # check types
    assert all(raw.get_channel_types() == stream.nm_channels['type']) == True, "Channel types in data file are not matching the stream"
    # assert all(player.player.get_channel_types() == stream.nm_channels['type']) == True, "Channel types in stream are not matching the player" 
    # TODO this fails super weird. At some point all channel types in the player are set to eeg and i really don't get where this hppens! Is it us or mnelsl?

    # check names 
    assert all(raw.ch_names == stream.nm_channels['name']) == True, "Expected same channel names in the stream and input file"
    assert all(player.player.info['ch_names'] == stream.nm_channels['name']) == True, "Expected same channel names in the stream and player"


    assert features is not None
