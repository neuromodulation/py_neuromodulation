import webbrowser

from py_neuromodulation.stream import (
    LSLStream,
    LSLOfflinePlayer,
)
from py_neuromodulation import io
from py_neuromodulation import App


def main():
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

    player = LSLOfflinePlayer(raw=raw, stream_name="example_stream")

    player.start_player(chunk_size=30, n_repeat=5999999)

    webbrowser.open_new_tab("http://localhost:50001")

    App(run_in_webview=False, dev=False).launch()

    

if __name__ == "__main__":
    main()
