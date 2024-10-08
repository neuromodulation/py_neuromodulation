import numpy as np
import mne
from pathlib import Path
import multiprocessing as mp
import atexit
import time
import signal

from py_neuromodulation.utils.types import _PathLike
from py_neuromodulation.utils.io import read_BIDS_data
from py_neuromodulation.utils import logger


class LSLOfflinePlayer:
    _instances: set["LSLOfflinePlayer"] = set()  # Keep track of initialized players
    _atexit_registered: bool = False  # Flag to register atexit

    def __init__(
        self,
        stream_name: str = "lsl_offline_player",
        f_name: str | _PathLike = None,
        raw: mne.io.Raw | None = None,
        sfreq: int | float | None = None,
        data: np.ndarray | None = None,
        ch_type: str | None = "dbs",
        chunk_size: int = 10,
        n_repeat: int = 1,
    ) -> None:
        """Initialization of MNE-LSL offline player.
        Either a filename (PathLike) is provided,
        or data and sampling frequency to initialize an example mock-up stream.


        Parameters
        ----------
        stream_name : str, optional
            LSL stream name, by default "example_stream"
        f_name : str | None, optional
            file name used for streaming, by default None
        sfreq : int | float | None, optional
            sampling rate, by default None
        data : np.ndarray | None, optional
            data used for streaming, by default None
        ch_type: str | None, optional
            channel type to select for streaming, by default "dbs"

        Raises
        ------
        ValueError
            _description_
        """
        self.sfreq = sfreq
        self.stream_name = stream_name
        self.chunk_size = chunk_size
        self.n_repeat = n_repeat

        if f_name:
            (self._path_raw, data, sfreq, line_noise, coord_list, coord_names) = (
                read_BIDS_data(f_name)
            )
        elif raw:
            self._path_raw = raw
        elif sfreq and data:
            info = mne.create_info(
                ch_names=[f"ch{i}" for i in range(data.shape[0])],
                ch_types=[ch_type for _ in range(data.shape[0])],
                sfreq=sfreq,
            )
            raw = mne.io.RawArray(data, info)
            self._path_raw = Path.cwd() / "temp_raw.fif"
            raw.save(self._path_raw, overwrite=True)
        else:
            error_msg = "Either f_name or raw or sfreq and data must be provided."
            logger.critical(error_msg)
            raise ValueError(error_msg)

        # Flags to control the player subprocess
        self._streaming_complete = mp.Event()
        self._player_process = None
        self._stop_flag = mp.Event()
        self._started_streaming = mp.Event()

        LSLOfflinePlayer._instances.add(self)  # Register instancwe
        if LSLOfflinePlayer._atexit_registered:
            atexit.register(LSLOfflinePlayer._stop_all_players)
            LSLOfflinePlayer._atexit_registered = True

    def start_player(
        self,
        chunk_size: int | None = None,
        n_repeat: int | None = None,
        block: bool = False,
    ):
        """Start MNE-LSL Player

        Parameters
        ----------
        chunk_size : int, optional
            Number of samples to stream at once, by default 10
        n_repeat : int, optional
            Number of times to repeat the stream, by default 1
        block : bool, optional
            If True, block until streaming is complete, by default False
        """

        if chunk_size:
            self.chunk_size = chunk_size
        if n_repeat:
            self.n_repeat = n_repeat

        self._stop_flag.clear()
        self._streaming_complete.clear()

        self._player_process = mp.Process(
            target=self._run_player,
            args=(
                self.chunk_size,
                self.n_repeat,
                self._stop_flag,
                self._streaming_complete,
                self._started_streaming,
            ),
        )
        self._player_process.start()
        while not self._started_streaming.is_set():
            time.sleep(0.1)

        if block:
            try:
                self.wait_for_completion()
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received. Stopping the player...")
                self.stop_player()

    def _run_player(self, chunk_size, n_repeat, stop_flag, streaming_complete, started_streaming):
        from mne_lsl.player import PlayerLSL

        signal.signal(signal.SIGINT, lambda: stop_flag.set())

        player = PlayerLSL(
            self._path_raw,
            name=self.stream_name,
            chunk_size=chunk_size,
            n_repeat=n_repeat,
        )
        player = player.start()
        started_streaming.set()

        try:
            while not stop_flag.is_set() and not player._end_streaming:
                time.sleep(0.1)
        finally:
            player.stop()
            streaming_complete.set()

    def wait_for_completion(self):
        """Block until streaming is complete"""
        while self._player_process and self._player_process.is_alive():
            try:
                self._streaming_complete.wait(timeout=1.0)
                if self._streaming_complete.is_set():
                    break
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received. Stopping the player...")
                self.stop_player()
                break

    def stop_player(self):
        """Stop MNE-LSL Player"""
        if self._player_process and self._player_process.is_alive():
            self._stop_flag.set()
            self._player_process.join(timeout=5)
            if self._player_process.is_alive():
                self._player_process.terminate()
                self._player_process.join(timeout=1)
            if self._player_process.is_alive():
                self._player_process.kill()
            self._player_process = None

        print(f"Player stopped: {self.stream_name}")
        LSLOfflinePlayer._instances.discard(self)

    @classmethod
    def _stop_all_players(cls):
        """Stop all player instances (used for atexit)"""
        for player in cls._instances:
            player.stop_player()

    # Enable use as a context manager
    def __enter__(self):
        self.start_player()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_player()
