from mne_lsl.stream import StreamLSL
import numpy as np
import pytest
import time


@pytest.mark.parametrize("setup_lsl_player", ["search"], indirect=True)
def test_lsl_stream_search(setup_lsl_player):
    from py_neuromodulation.stream import mnelsl_stream

    """ Test if the lsl stream search can find any streams after starting a player."""
    player = setup_lsl_player
    player.start_player()
    streams = mnelsl_stream.resolve_streams()
    assert len(streams) != 0, "No streams found in search"


@pytest.mark.parametrize("setup_lsl_player", ["offline_test"], indirect=True)
def test_offline_lsl(
    setup_default_stream_fast_compute, setup_lsl_player, setup_default_data
):
    """ " Test the offline lsl player and stream, checking for sfreq, channel types and channel names."""

    raw, data, sfreq = setup_default_data
    player = setup_lsl_player
    player.start_player()

    data, stream = setup_default_stream_fast_compute

    features = stream.run(
        is_stream_lsl=True,
        plot_lsl=False,
        stream_lsl_name="offline_test",
        out_path_root="./test_data",
        folder_name="test_offline_lsl",
    )
    # check sfreq
    if not raw.info["sfreq"] == stream.sfreq:
        raise ValueError(
            "Expected same sampling frequency in the stream and input file"
        )
    if not player.player.info["sfreq"] == stream.sfreq:
        raise ValueError("Expected same sampling frequency in the stream and player")

    # check types
    if not all(raw.get_channel_types() == stream.channels["type"]):
        raise ValueError("Channel types in data file are not matching the stream")
    if not all(player.player.get_channel_types() == stream.channels["type"]):
        raise ValueError("Channel types in stream are not matching the player")

    # check names
    if not all(raw.ch_names == stream.channels["name"]):
        raise ValueError("Expected same channel names in the stream and input file")
    if not all(player.player.info["ch_names"] == stream.channels["name"]):
        raise ValueError("Expected same channel names in the stream and player")

    if not features is not None:
        raise ValueError(
            "Expected features dataframe as stream.run() output but got None"
        )


@pytest.mark.parametrize("setup_lsl_player", ["data_test"], indirect=True)
def test_lsl_data(setup_default_data, setup_lsl_player):
    """Check if 99% of the data is the same in the stream and the raw data."""
    import pandas as pd

    raw, data, sfreq = setup_default_data
    df_stream = pd.DataFrame()
    player = setup_lsl_player
    player.start_player(chunk_size=2)
    stream_player_check = StreamLSL(bufsize=2, name="data_test").connect()
    time.sleep(0.2)
    while stream_player_check.n_new_samples != 0:
        winsize = stream_player_check.n_new_samples / stream_player_check.info["sfreq"]
        data, ts = stream_player_check.get_data(winsize)
        df_stream = pd.concat([df_stream, pd.DataFrame(data)], axis=1)
        time.sleep(0.5)

    raw_sliced = pd.DataFrame(raw.get_data()).iloc[:, -df_stream.shape[1] :]
    df_stream.columns = range(len(df_stream.columns))
    raw_sliced_values = raw_sliced.values
    df_stream_values = df_stream.values
    same_values = np.zeros(raw_sliced.shape[1], dtype=bool)
    for i in range(raw_sliced.shape[1]):
        same_values[i] = np.any(
            np.all(raw_sliced_values == df_stream_values[:, i][:, np.newaxis], axis=0)
        )
    matching_percentage = np.sum(same_values) / len(same_values) * 100

    assert np.any(
        matching_percentage >= 99
    ), f"Expected same data in at least 99 percent of the samples but got {np.max(matching_percentage)} percent"
