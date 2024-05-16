from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL
import numpy as np
import time
from py_neuromodulation import (nm_mnelsl_generator, nm_mnelsl_stream, nm_IO, nm_settings)
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
    
def test_mne_lsl():

    player1 = PlayerLSL(raw,  name="general_lsl_test_stream", chunk_size=10)
    player1 = player1.start()

    stream1 = StreamLSL(name="general_lsl_test_stream", bufsize=2).connect()
    ch_types = stream1.get_channel_types(unique=True)
    assert 'dbs' in ch_types, "Expected at least one dbs channel from example data"
    assert player1.info['nchan'] == 10, "Expected 10 channels from example data"
    data_l = []
    timestamps_l = []

    def call_every_100ms():
        data, timestamps = stream1.get_data(winsize=10)
        data_l.append(data)
        timestamps_l.append(timestamps)

    t = threading.Timer(0.1, call_every_100ms)
    t.start()

    # import time

    time_start = time.time()

    while time.time() - time_start <= 10:
        time.sleep(1)
    t.cancel()

    collected_data_shape = np.concatenate(data_l).shape
    assert collected_data_shape[0] > 0 and collected_data_shape[1] > 0, "Expected non-empty data"

    stream1.disconnect()
    player1.stop()


def test_lsl_stream_search():
    player = nm_mnelsl_generator.LSLOfflinePlayer(f_name = raw)
    player.start_player()
    streams = nm_mnelsl_stream.resolve_streams()
    assert len(streams) != 0, "No streams found in search"


def test_offline_lsl(setup_default_stream_fast_compute):
    settings = nm_settings.get_default_settings()
    settings = nm_settings.set_settings_fast_compute(settings)

    player = nm_mnelsl_generator.LSLOfflinePlayer(f_name = raw)
    player.start_player()

    data, stream = setup_default_stream_fast_compute

    features = stream.run(stream_lsl = True, plot_lsl= False)
    # check sfreq
    assert raw.info['sfreq'] == stream.sfreq, "Expected same sampling frequency in the stream and input file"
    assert player.player.info['sfreq'] == stream.sfreq, "Expected same sampling frequency in the stream and player"

    # check types
    assert all(raw.get_channel_types() == stream.nm_channels['type']) == True, "Channel types in data file are not matching the stream"
    assert all(player.player.get_channel_types() == stream.nm_channels['type']) == True, "Channel types in stream are not matching the player" 

    # check names 
    assert all(raw.ch_names == stream.nm_channels['name']) == True, "Expected same channel names in the stream and input file"
    assert all(player.player.info['ch_names'] == stream.nm_channels['name']) == True, "Expected same channel names in the stream and player"


    assert features is not None


def test_lsl_data(setup_default_stream_fast_compute):
    import pandas as pd
    data_l = pd.DataFrame()
    player = nm_mnelsl_generator.LSLOfflinePlayer(f_name=raw, stream_name="data_test_stream")
    player.start_player(chunk_size=1)
    # data, stream = setup_default_stream_fast_compute
    stream_player_check = StreamLSL(name="data_test_stream", bufsize=2).connect()
    time.sleep(0.5)
    winsize = stream_player_check.n_new_samples / stream_player_check.info["sfreq"]
    while(stream_player_check.n_new_samples != 0):
        winsize = stream_player_check.n_new_samples / stream_player_check.info["sfreq"]
        data, ts = stream_player_check.get_data(winsize)
        data_l = pd.concat([data_l, pd.DataFrame(data)],axis=1)
        time.sleep(0.5)

    raw_equals_player = []
    raw_sliced = pd.DataFrame(raw.get_data()).iloc[:, -data_l.shape[1]:]
    data_l.columns = range(len(data_l.columns))
    data_l_array = data_l.to_numpy()
    raw_sliced_array = raw_sliced.to_numpy()


    data_l_reshaped = data_l_array[:, np.newaxis, :]

    raw_row_equals_player = np.all(data_l_reshaped.transpose() == raw_sliced_array[:, np.newaxis].transpose(), axis=(2, 1))
    true_counts = np.sum(raw_row_equals_player, axis=0)
    matching_percentage = (true_counts / raw_row_equals_player.shape[0]) * 100

    raw_equals_player = np.all(data_l_array == raw_sliced_array, axis=0)
   
    # testing if at least 10% of rows match
    assert np.any(matching_percentage >= 10), f"Expected same data in at least 10 percent of the samples but got {np.max(matching_percentage)} percent"

    # 'testing if at least 10% of the data matches exactly, not allowing for index shifts'
    # assert np.count_nonzero(raw_equals_player) >= (len(raw_equals_player)/10), f"Expected same data in at least 10 percent of the samples but got {(np.count_nonzero(raw_equals_player)/len(raw_equals_player))*100} percent"
