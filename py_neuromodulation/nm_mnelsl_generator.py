from mne_lsl.player import PlayerLSL
import numpy as np
import mne
from pathlib import Path

from py_neuromodulation import logger, nm_types


class LSLOfflinePlayer:

    def __init__(
        self,
        stream_name: str | None = "lsl_offline_player",
        f_name: str | nm_types.PathLike = None,
        sfreq: int | float | None = None,
        data: np.ndarray | None = None,
        ch_types: str | None = "dbs",
    ) -> None:
        """Initialization of MNE-LSL offline player.
        Either a filename (PathLike) is provided,
        or data and sampling frequency to initialize an example mock-up stream.


        Parameters
        ----------
        settings : dict
            _description_
        stream_name : str, optional
            _description_, by default "example_stream"
        f_name : str | None, optional
            _description_, by default None
        sfreq : int | float | None, optional
            _description_, by default None
        data : np.ndarray | None, optional
            _description_, by default None

        Raises
        ------
        ValueError
            _description_
        """
        self.sfreq = sfreq
        self.stream_name = stream_name
        got_fname = f_name is not None
        got_sfreq_data = sfreq is not None and data is not None
        if not (got_fname or got_sfreq_data):
            error_msg = "Either f_name or sfreq and data must be provided."
            logger.critical(error_msg)
            raise ValueError(error_msg)

        if f_name is not None:
            self._path_raw = f_name

        if sfreq is not None and data is not None:

            info = mne.create_info(
                ch_names=[f"ch{i}" for i in range(data.shape[0])],
                ch_types=[ch_types for _ in range(data.shape[0])],
                sfreq=sfreq,
            )
            raw = mne.io.RawArray(data, info)
            self._path_raw = Path.cwd() / "temp_raw.fif"
            raw.save(self._path_raw, overwrite=True)

    def start_player(self, chunk_size: int = 1, n_repeat: int = 1):
        """Start MNE-LSL Player

        Parameters
        ----------
        chunk_size : int, optional
            _description_, by default 1
        n_repeat : int, optional
            _description_, by default 1
        """
        self.player = PlayerLSL(
            self._path_raw,
            name=self.stream_name,
            chunk_size=chunk_size,
            n_repeat=n_repeat,
        )
        self.player = self.player.start()

    def stop_player(self):
        """Stop MNE-LSL Player"""
        self.player.stop()
