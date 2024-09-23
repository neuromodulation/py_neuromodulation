"""Module for generic and offline data streams."""

from typing import TYPE_CHECKING
from collections.abc import Iterator
import numpy as np
import pandas as pd
from pathlib import Path

import multiprocessing as mp
from contextlib import suppress

from py_neuromodulation.stream.data_processor import DataProcessor
from py_neuromodulation.utils.io import MNE_FORMATS, read_mne_data
from py_neuromodulation.utils.types import _PathLike
from py_neuromodulation.stream.settings import NMSettings
from py_neuromodulation.features import USE_FREQ_RANGES
from py_neuromodulation.utils import (
    logger,
    create_default_channels_from_data,
    load_channels,
    save_features,
    create_channels,
)
from py_neuromodulation.gui.backend.app_socket import WebSocketManager
from py_neuromodulation import PYNM_DIR

if TYPE_CHECKING:
    import pandas as pd


class Stream:
    """_GenericStream base class.
    This class can be inherited for different types of offline streams

    Parameters
    ----------
    nm_stream_abc : stream_abc.NMStream
    """

    def __init__(
        self,
        data: "np.ndarray | pd.DataFrame | _PathLike | None" = None,
        sfreq: float | None = None,
        experiment_name: str = "sub",
        channels: "pd.DataFrame | _PathLike | None" = None,
        is_stream_lsl: bool = False,
        stream_lsl_name: str | None = None,
        settings: NMSettings | _PathLike | None = None,
        line_noise: float | None = 50,
        sampling_rate_features_hz: float | None = None,
        path_grids: _PathLike | None = None,
        coord_names: list | None = None,
        coord_list: list | None = None,
        verbose: bool = True,
    ) -> None:
        """Stream initialization

        Parameters
        ----------
        sfreq : float
            sampling frequency of data in Hertz
        channels : pd.DataFrame | _PathLike
            parametrization of channels (see define_channels.py for initialization)
        data : np.ndarray | pd.DataFrame | None, optional
            data to be streamed with shape (n_channels, n_time), by default None
        settings : NMSettings | _PathLike | None, optional
            Initialized settings.NMSettings object, by default the py_neuromodulation/settings.yaml are read
            and passed into a settings object
        line_noise : float | None, optional
            line noise, by default 50
        sampling_rate_features_hz : float | None, optional
            feature sampling rate, by default None
        path_grids : _PathLike | None, optional
            path to grid_cortex.tsv and/or gird_subcortex.tsv, by default Non
        coord_names : list | None, optional
            coordinate name in the form [coord_1_name, coord_2_name, etc], by default None
        coord_list : list | None, optional
            coordinates in the form [[coord_1_x, coord_1_y, coord_1_z], [coord_2_x, coord_2_y, coord_2_z],], by default None
        verbose : bool, optional
            print out stream computation time information, by default True
        """
        # Input params
        self.path_grids = path_grids
        self.verbose = verbose
        self.line_noise = line_noise
        self.coord_names = coord_names
        self.coord_list = coord_list
        self.experiment_name = experiment_name
        self.data = data
        self.settings: NMSettings = NMSettings.load(settings)
        self.is_stream_lsl = is_stream_lsl
        self.stream_lsl_name = stream_lsl_name

        self.sess_right = None
        self.projection = None
        self.model = None

        if sampling_rate_features_hz is not None:
            self.settings.sampling_rate_features_hz = sampling_rate_features_hz

        if path_grids is None:
            path_grids = PYNM_DIR

        # Set up some flags for stream processing later
        self.is_running = False
        self.target_idx_initialized: bool = False

        # Validate input depending on stream type and initialize stream
        self.generator: Iterator

        if self.is_stream_lsl:
            from py_neuromodulation.stream.mnelsl_stream import LSLStream

            if self.stream_lsl_name is None:
                logger.info(
                    "No stream name specified. Will connect to the first available stream if it exists."
                )

            print(self.stream_lsl_name)
            self.lsl_stream = LSLStream(
                settings=self.settings, stream_name=self.stream_lsl_name
            )

            sinfo = self.lsl_stream.sinfo

            # If no sampling frequency is specified in the stream, try to get it from the passed parameters
            if sinfo.sfreq is None:
                logger.info("No sampling frequency specified in LSL stream")
                if sfreq is not None:
                    logger.info("Using sampling frequency passed to Stream constructor")
                else:
                    raise ValueError(
                        "No sampling frequency specified in stream and no sampling frequency passed to Stream constructor"
                    )
            else:
                if sfreq is not None != sinfo.sfreq:
                    logger.info(
                        "Sampling frequency of the LSL stream does not match the passed sampling frequency."
                    )
                logger.info("Using sampling frequency of the LSL stream")
                self.sfreq = sinfo.sfreq

            # TONI: should we try to get channels from the passed "channels" parameter before generating default?

            # Try to get channel names and types from the stream, if not generate default
            ch_names = sinfo.get_channel_names() or [
                "ch" + str(i) for i in range(sinfo.n_channels)
            ]
            ch_types = sinfo.get_channel_types() or [
                "eeg" for i in range(sinfo.n_channels)
            ]
            self.channels = create_channels(
                ch_names=ch_names,
                ch_types=ch_types,
                used_types=["eeg", "ecog", "dbs", "seeg"],
            )

            self.generator = self.lsl_stream.get_next_batch()

        else:  # Data passed as array, dataframe or path to file
            if data is None:
                raise ValueError(
                    "If is_stream_lsl is False, data must be passed to the Stream constructor"
                )

            # If channels passed to constructor, try to load them
            self.channels = load_channels(channels) if channels is not None else None

            if isinstance(self.data, (np.ndarray, pd.DataFrame)):
                logger.info(f"Loading data from {type(data).__name__}")

                if sfreq is None:
                    raise ValueError(
                        "sfreq must be specified when passing data as an array or dataframe"
                    )

                self.sfreq = sfreq

                if self.channels is None:
                    self.channels = create_default_channels_from_data(self.data)

                self.data = self._handle_data(self.data)

            elif isinstance(self.data, _PathLike):
                # If data is a path, try to load it as an MNE supported file
                logger.info("Loading data from file")
                filepath = Path(self.data)  # type: ignore
                ext = filepath.suffix

                if ext in MNE_FORMATS:
                    data, sfreq, ch_names, ch_types, bads = read_mne_data(filepath)
                else:
                    raise ValueError(f"Unsupported file format: {ext}")

                if sfreq is None:
                    raise ValueError(
                        "Sampling frequency not specified in file, please specify sfreq as a parameters"
                    )

                self.sfreq = sfreq

                self.channels = create_channels(
                    ch_names=ch_names,
                    ch_types=ch_types,
                    used_types=["eeg", "ecog", "dbs", "seeg"],
                    bads=bads,
                )

                # _handle_data requires the channels to be set
                self.data = self._handle_data(data)

            else:
                raise ValueError(
                    "Data must be either a numpy array, a pandas DataFrame, or a path to an MNE supported file"
                )

            from py_neuromodulation.stream.generator import RawDataGenerator

            self.generator: Iterator = RawDataGenerator(
                self.data,
                self.sfreq,
                self.settings.sampling_rate_features_hz,
                self.settings.segment_length_features_ms,
            )

        self._initialize_data_processor()

    def _add_target(self, feature_dict: dict, data: np.ndarray) -> None:
        """Add target channels to feature series.

        Parameters
        ----------
        feature_dict : dict
        data : np.ndarray
            Raw data with shape (n_channels, n_samples).
            Channels not usd for feature computation are also included

        Returns
        -------
        dict
            feature dict with target channels added
        """
        if not (isinstance(self.channels, pd.DataFrame)):
            raise ValueError("Channels must be a pandas DataFrame")

        if self.channels["target"].sum() > 0:
            if not self.target_idx_initialized:
                self.target_indexes = self.channels[self.channels["target"] == 1].index
                self.target_names = self.channels.loc[
                    self.target_indexes, "name"
                ].to_list()
                self.target_idx_initialized = True

            for target_idx, target_name in zip(self.target_indexes, self.target_names):
                feature_dict[target_name] = data[target_idx, -1]

    def run(
        self,
        out_dir: _PathLike = "",
        save_csv: bool = False,
        save_interval: int = 10,
        return_df: bool = True,
        stream_handling_queue: "mp.Queue | None" = None,
        websocket_featues: WebSocketManager | None = None,
    ):
        # Check that at least one channel is selected for analysis
        if self.channels.query("used == 1 and target == 0").shape[0] == 0:
            raise ValueError(
                "No channels selected for analysis that have column 'used' = 1 and 'target' = 0. Please check your channels"
            )

        # If features that use frequency ranges are on, test them against nyquist frequency
        need_nyquist_check = any(
            (f in USE_FREQ_RANGES for f in self.settings.features.get_enabled())
        )

        if need_nyquist_check:
            assert all(
                fb.frequency_high_hz < self.sfreq / 2
                for fb in self.settings.frequency_ranges_hz.values()
            ), (
                "If a feature that uses frequency ranges is selected, "
                "the frequency band ranges need to be smaller than the nyquist frequency.\n"
                f"Got sfreq = {self.sfreq} and fband ranges:\n {self.settings.frequency_ranges_hz}"
            )

        self.stream_handling_queue = stream_handling_queue
        # self.feature_queue = feature_queue
        self.save_csv = save_csv
        self.save_interval = save_interval
        self.return_df = return_df

        # Generate output dirs
        self.out_dir_root = Path.cwd() if not out_dir else Path(out_dir)
        self.out_dir = self.out_dir_root / self.experiment_name
        # TONI: Need better default experiment name

        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Open database connection
        # TONI: we should give the user control over the save format
        from py_neuromodulation.utils.database import NMDatabase

        self.db = NMDatabase(self.experiment_name, out_dir)  # Create output database

        self.batch_count: int = 0  # Keep track of the number of batches processed

        # Reinitialize the data processor in case the nm_channels or nm_settings changed between runs of the same Stream
        self._initialize_data_processor()

        logger.log_to_file(out_dir)

        # # Initialize mp.Pool for multiprocessing
        # self.pool = mp.Pool(processes=self.settings.n_jobs)
        # # Set up shared memory for multiprocessing
        # self.shared_memory = mp.Array(ctypes.c_double, self.settings.n_jobs * self.settings.n_jobs)
        # # Set up multiprocessing semaphores
        # self.semaphore = mp.Semaphore(self.settings.n_jobs)

        prev_batch_end = 0
        for timestamps, data_batch in self.generator:
            self.is_running = True
            if self.stream_handling_queue is not None:
                if not self.stream_handling_queue.empty():
                    value = self.stream_handling_queue.get()
                    if value == "stop":
                        break
            if data_batch is None:
                break

            feature_dict = self.data_processor.process(data_batch)

            this_batch_end = timestamps[-1]
            batch_length = this_batch_end - prev_batch_end
            logger.debug(
                f"{batch_length:.3f} seconds of new data processed",
            )

            feature_dict["time"] = (
                batch_length
                if self.is_stream_lsl
                else np.ceil(this_batch_end * 1000 + 1)
            )

            prev_batch_end = this_batch_end

            if self.verbose:
                logger.info("Time: %.2f", feature_dict["time"] / 1000)

            self._add_target(feature_dict, data_batch)

            # We should ensure that feature output is always either float64 or None and remove this
            with suppress(TypeError):  # Need this because some features output None
                for key, value in feature_dict.items():
                    feature_dict[key] = np.float64(value)

            self.db.insert_data(feature_dict)

            # if self.feature_queue is not None:
            #    self.feature_queue.put(feature_dict)

            # if websocket_features is not None:
            #     logger.info("Sending message to Websocket")
            #     await websocket_featues.send_message(feature_dict)

            self.batch_count += 1
            if self.batch_count % self.save_interval == 0:
                self.db.commit()

        self.db.commit()  # Save last batches

        # If save_csv is False, still save the first row to get the column names
        feature_df: "pd.DataFrame" = (
            self.db.fetch_all() if (self.save_csv or self.return_df) else self.db.head()
        )

        self.db.close()  # Close the database connection

        self._save_after_stream(feature_arr=feature_df)
        self.is_running = False

        return feature_df  # TONI: Not sure if this makes sense anymore

    def plot_raw_signal(
        self,
        sfreq: float | None = None,
        data: np.ndarray | None = None,
        lowpass: float | None = None,
        highpass: float | None = None,
        picks: list | None = None,
        plot_time: bool = True,
        plot_psd: bool = False,
    ) -> None:
        """Use MNE-RawArray Plot to investigate PSD or raw_signal plot.

        Parameters
        ----------
        sfreq : float
            sampling frequency [Hz]
        data : np.ndarray, optional
            data (n_channels, n_times), by default None
        lowpass: float, optional
            cutoff lowpass filter frequency
        highpass: float, optional
            cutoff highpass filter frequency
        picks: list, optional
            list of channels to plot
        plot_time : bool, optional
            mne.io.RawArray.plot(), by default True
        plot_psd : bool, optional
            mne.io.RawArray.plot(), by default False

        Raises
        ------
        ValueError
            raise Exception when no data is passed
        """
        if self.data is None and data is None:
            raise ValueError("No data passed to plot_raw_signal function.")

        if data is None and self.data is not None:
            data = self.data

        if sfreq is None:
            sfreq = self.sfreq

        if self.channels is not None:
            ch_names = self.channels["name"].to_list()
            ch_types = self.channels["type"].to_list()
        else:
            ch_names = [f"ch_{i}" for i in range(data.shape[0])]
            ch_types = ["ecog" for i in range(data.shape[0])]

        from mne import create_info
        from mne.io import RawArray

        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = RawArray(data, info)

        if picks is not None:
            raw = raw.pick(picks)
        self.raw = raw
        if plot_time:
            raw.plot(highpass=highpass, lowpass=lowpass)
        if plot_psd:
            raw.compute_psd().plot()

    def _handle_data(self, data: "np.ndarray | pd.DataFrame") -> np.ndarray:
        """_summary_

        Args:
            data (np.ndarray | pd.DataFrame):
        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            np.ndarray: _description_
        """
        names_expected = self.channels["name"].to_list()

        if isinstance(data, np.ndarray):
            if not len(names_expected) == data.shape[0]:
                raise ValueError(
                    "If data is passed as an array, the first dimension must"
                    " match the number of channel names in `channels`.\n"
                    f" Number of data channels (data.shape[0]): {data.shape[0]}\n"
                    f' Length of channels["name"]: {len(names_expected)}.'
                )
            return data

        names_data = data.columns.to_list()

        if not (
            len(names_expected) == len(names_data)
            and sorted(names_expected) == sorted(names_data)
        ):
            raise ValueError(
                "If data is passed as a DataFrame, the"
                "column names must match the channel names in `channels`.\n"
                f"Input dataframe column names: {names_data}\n"
                f'Expected (from channels["name"]): : {names_expected}.'
            )
        return data.to_numpy().transpose()

    def _initialize_data_processor(self) -> None:
        self.data_processor = DataProcessor(
            sfreq=self.sfreq,
            settings=self.settings,
            channels=self.channels,
            path_grids=self.path_grids,
            coord_names=self.coord_names,
            coord_list=self.coord_list,
            line_noise=self.line_noise,
            verbose=self.verbose,
        )

    def _save_after_stream(
        self,
        feature_arr: "pd.DataFrame | None" = None,
    ) -> None:
        """Save features, settings, nm_channels and sidecar after run"""
        self._save_sidecar()
        if feature_arr is not None:
            self._save_features(feature_arr)
        self._save_settings()
        self._save_channels()

    def _save_features(
        self,
        feature_arr: "pd.DataFrame",
    ) -> None:
        save_features(feature_arr, self.out_dir, self.experiment_name)

    def _save_channels(self) -> None:
        self.data_processor.save_channels(self.out_dir, self.experiment_name)

    def _save_settings(self) -> None:
        self.data_processor.save_settings(self.out_dir, self.experiment_name)

    def _save_sidecar(self) -> None:
        """Save sidecar incduing fs, coords, sess_right to
        out_path_root and subfolder 'folder_name'"""
        additional_args = {"sess_right": self.sess_right}
        self.data_processor.save_sidecar(
            self.out_dir, self.experiment_name, additional_args
        )
