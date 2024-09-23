from py_neuromodulation.stream import (
    LSLStream,
    LSLOfflinePlayer,
)
import os
import asyncio

from py_neuromodulation.utils import create_channels

from py_neuromodulation import io

from py_neuromodulation import App
import py_neuromodulation as nm
import numpy as np
import multiprocessing as mp

(
    RUN_NAME,
    PATH_RUN,
    PATH_BIDS,
    PATH_OUT,
    datatype,
) = io.get_paths_example_data()

(
    raw,
    data,
    sfreq,
    line_noise,
    coord_list,
    coord_names,
) = io.read_BIDS_data(PATH_RUN=PATH_RUN)

if __name__ == "__main__":
    # PATH_VHDR = "/Users/Timon/Documents/py-neurmodulation_merge/py_neuromodulation/py_neuromodulation/data/sub-testsub/ses-EphysMedOff/ieeg/sub-testsub_ses-EphysMedOff_task-gripforce_run-0_ieeg.vhdr"

    # data, sfreq, ch_names, ch_types, bads = io.read_mne_data(PATH_VHDR)

    # channels = set_channels(
    #     ch_names=ch_names,
    #     ch_types=ch_types,
    #     bads=bads,
    #     reference=None,
    #     used_types=["eeg", "ecog", "dbs", "seeg"],
    #     target_keywords=None,
    # )

    # (
    #     raw_arr,
    #     data,
    #     sfreq,
    #     line_noise,
    #     coord_list,
    #     coord_names,
    # ) = io.read_BIDS_data(PATH_RUN=PATH_RUN)

    # channels = set_channels(
    #     ch_names=raw_arr.ch_names,
    #     ch_types=raw_arr.get_channel_types(),
    #     bads=None,
    #     reference=None,
    #     used_types=["eeg", "ecog", "dbs", "seeg"],
    # )

    # settings = nm.NMSettings.get_fast_compute()

    # stream = nm.Stream(
    #     settings=settings,
    #     channels=channels,
    #     verbose=True,
    #     sfreq=sfreq,
    #     line_noise=50,
    # )

    # features = asyncio.run(stream.run(data, save_csv=True))

    player = LSLOfflinePlayer(raw=raw, stream_name="example_stream")

    player.start_player(chunk_size=30, n_repeat=5999999)

    App(run_in_webview=False).launch()

    # check functionality of stream: do features end up in the queue?

    # are queue values put through websocket
