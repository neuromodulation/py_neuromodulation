"""Module for generic and offline data streams."""

from typing import TYPE_CHECKING
from collections.abc import Iterator
import numpy as np
from pathlib import Path

import py_neuromodulation as nm
from contextlib import suppress

from py_neuromodulation.stream.data_processor import DataProcessor
from py_neuromodulation.utils.types import _PathLike, FeatureName
from py_neuromodulation.stream.settings import NMSettings

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
        sfreq: float,
        channels: "pd.DataFrame | _PathLike | None" = None,
        data: "np.ndarray | pd.DataFrame | None" = None,
        settings: NMSettings | _PathLike | None = None,
        line_noise: float | None = 50,
        sampling_rate_features_hz: float | None = None,
        path_grids: _PathLike | None = None,
        coord_names: list | None = None,
        stream_name: str
        | None = "example_stream",  # Timon: do we need those in the nmstream_abc?
        is_stream_lsl: bool = False,
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
        self.settings: NMSettings = NMSettings.load(settings)

        if channels is None and data is not None:
            channels = nm.utils.channels.get_default_channels_from_data(data)

        if channels is not None:
            self.channels = nm.io.load_channels(channels)

        if self.channels.query("used == 1 and target == 0").shape[0] == 0:
            raise ValueError(
                "No channels selected for analysis that have column 'used' = 1 and 'target' = 0. Please check your channels"
            )

        if channels is None and data is None:
            raise ValueError("Either `channels` or `data` must be passed to `Stream`.")

        # If features that use frequency ranges are on, test them against nyquist frequency
        use_freq_ranges: list[FeatureName] = [
            "bandpass_filter",
            "stft",
            "fft",
            "welch",
            "bursts",
            "coherence",
            "nolds",
            "bispectrum",
        ]

        need_nyquist_check = any(
            (f in use_freq_ranges for f in self.settings.features.get_enabled())
        )

        if need_nyquist_check:
            assert all(
                fb.frequency_high_hz < sfreq / 2
                for fb in self.settings.frequency_ranges_hz.values()
            ), (
                "If a feature that uses frequency ranges is selected, "
                "the frequency band ranges need to be smaller than the nyquist frequency.\n"
                f"Got sfreq = {sfreq} and fband ranges:\n {self.settings.frequency_ranges_hz}"
            )

        if sampling_rate_features_hz is not None:
            self.settings.sampling_rate_features_hz = sampling_rate_features_hz

        if path_grids is None:
            path_grids = nm.PYNM_DIR

        self.path_grids = path_grids
        self.verbose = verbose
        self.sfreq = sfreq
        self.line_noise = line_noise
        self.coord_names = coord_names
        self.coord_list = coord_list
        self.sess_right = None
        self.projection = None
        self.model = None

        # TODO(toni): is it necessary to initialize the DataProcessor on stream init?
        self.data_processor = DataProcessor(
            sfreq=self.sfreq,
            settings=self.settings,
            channels=self.channels,
            path_grids=self.path_grids,
            coord_names=coord_names,
            coord_list=coord_list,
            line_noise=line_noise,
            verbose=self.verbose,
        )

        self.data = data

        self.target_idx_initialized: bool = False

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

        if self.channels["target"].sum() > 0:
            if not self.target_idx_initialized:
                self.target_indexes = self.channels[self.channels["target"] == 1].index
                self.target_names = self.channels.loc[
                    self.target_indexes, "name"
                ].to_list()
                self.target_idx_initialized = True

            for target_idx, target_name in zip(self.target_indexes, self.target_names):
                feature_dict[target_name] = data[target_idx, -1]

    def _handle_data(self, data: "np.ndarray | pd.DataFrame") -> np.ndarray:
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

    def run(
        self,
        data: "np.ndarray | pd.DataFrame | None" = None,
        out_dir: _PathLike = "",
        experiment_name: str = "sub",
        is_stream_lsl: bool = False,
        stream_lsl_name: str | None = None,
        plot_lsl: bool = False,
        save_csv: bool = False,
        save_interval: int = 10,
        return_df: bool = True,
    ) -> "pd.DataFrame":
        self.is_stream_lsl = is_stream_lsl
        self.stream_lsl_name = stream_lsl_name

        # Validate input data
        if data is not None:
            data = self._handle_data(data)
        elif self.data is not None:
            data = self._handle_data(self.data)
        elif self.data is None and data is None and self.is_stream_lsl is False:
            raise ValueError("No data passed to run function.")

        # Generate output dirs
        self.out_dir_root = Path.cwd() if not out_dir else Path(out_dir)
        self.out_dir = self.out_dir_root / experiment_name
        # TONI: Need better default experiment name
        self.experiment_name = experiment_name if experiment_name else "sub"

        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Open database connection
        # TONI: we should give the user control over the save format
        from py_neuromodulation.utils.database import NMDatabase

        db = NMDatabase(experiment_name, out_dir)  # Create output database

        self.batch_count: int = 0  # Keep track of the number of batches processed

        # Reinitialize the data processor in case the nm_channels or nm_settings changed between runs of the same Stream
        # TONI: then I think we can just not initialize the data processor in the init function
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

        nm.logger.log_to_file(out_dir)

        # Initialize generator
        self.generator: Iterator
        if not is_stream_lsl:
            from py_neuromodulation.stream.generator import RawDataGenerator

            self.generator = RawDataGenerator(
                data,
                self.sfreq,
                self.settings.sampling_rate_features_hz,
                self.settings.segment_length_features_ms,
            )
        else:
            from py_neuromodulation.stream.mnelsl_stream import LSLStream

            self.lsl_stream = LSLStream(
                settings=self.settings, stream_name=stream_lsl_name
            )

            if plot_lsl:
                from mne_lsl.stream_viewer import StreamViewer

                viewer = StreamViewer(stream_name=stream_lsl_name)
                viewer.start()

            if self.sfreq != self.lsl_stream.stream.sinfo.sfreq:
                error_msg = (
                    f"Sampling frequency of the lsl-stream ({self.lsl_stream.stream.sinfo.sfreq}) "
                    f"does not match the settings ({self.sfreq})."
                    "The sampling frequency read from the stream will be used"
                )
                nm.logger.warning(error_msg)
                self.sfreq = self.lsl_stream.stream.sinfo.sfreq

            self.generator = self.lsl_stream.get_next_batch()

        prev_batch_end = 0
        for timestamps, data_batch in self.generator:
            if data_batch is None:
                break

            feature_dict = self.data_processor.process(data_batch)

            this_batch_end = timestamps[-1]
            batch_length = this_batch_end - prev_batch_end
            nm.logger.debug(
                f"{batch_length:.3f} seconds of new data processed",
            )

            feature_dict["time"] = (
                batch_length if is_stream_lsl else np.ceil(this_batch_end * 1000 + 1)
            )

            prev_batch_end = this_batch_end

            if self.verbose:
                nm.logger.info("Time: %.2f", feature_dict["time"] / 1000)

            self._add_target(feature_dict, data_batch)

            # We should ensure that feature output is always either float64 or None and remove this
            with suppress(TypeError):  # Need this because some features output None
                for key, value in feature_dict.items():
                    feature_dict[key] = np.float64(value)

            db.insert_data(feature_dict)

            self.batch_count += 1
            if self.batch_count % save_interval == 0:
                db.commit()

        db.commit()  # Save last batches

        # If save_csv is False, still save the first row to get the column names
        feature_df: "pd.DataFrame" = (
            db.fetch_all() if (save_csv or return_df) else db.head()
        )

        db.close()  # Close the database connection

        self._save_after_stream(feature_arr=feature_df)

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
        nm.io.save_features(feature_arr, self.out_dir, self.experiment_name)

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
