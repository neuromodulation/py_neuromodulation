from py_neuromodulation.stream import (
    LSLStream,
    LSLOfflinePlayer,
)

from py_neuromodulation import io

from py_neuromodulation import App
import numpy as np

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
    player = LSLOfflinePlayer(raw=raw, stream_name="example_stream")

    player.start_player(chunk_size=30, n_repeat=5999999)

    App(run_in_webview=False).launch()
