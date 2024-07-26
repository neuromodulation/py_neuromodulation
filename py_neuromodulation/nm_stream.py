"""Module for offline data streams."""

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from pathlib import Path
from py_neuromodulation.nm_stream_abc import NMStream
from py_neuromodulation.nm_types import _PathLike
from py_neuromodulation import logger

if TYPE_CHECKING:
    from py_neuromodulation.nm_settings import NMSettings


class _GenericStream(NMStream):
    """_GenericStream base class.
    This class can be inherited for different types of offline streams

    Parameters
    ----------
    nm_stream_abc : nm_stream_abc.NMStream
    """

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

        if self.nm_channels["target"].sum() > 0:
            if not self.target_idx_initialized:
                self.target_indexes = self.nm_channels[
                    self.nm_channels["target"] == 1
                ].index
                self.target_names = self.nm_channels.loc[
                    self.target_indexes, "name"
                ].to_list()
                self.target_idx_initialized = True

            for target_idx, target_name in zip(self.target_indexes, self.target_names):
                feature_dict[target_name] = data[target_idx, -1]

    def _handle_data(self, data: np.ndarray | pd.DataFrame) -> np.ndarray:
        names_expected = self.nm_channels["name"].to_list()

        if isinstance(data, np.ndarray):
            if not len(names_expected) == data.shape[0]:
                raise ValueError(
                    "If data is passed as an array, the first dimension must"
                    " match the number of channel names in `nm_channels`.\n"
                    f" Number of data channels (data.shape[0]): {data.shape[0]}\n"
                    f' Length of nm_channels["name"]: {len(names_expected)}.'
                )
            return data

        names_data = data.columns.to_list()
        if not (
            len(names_expected) == len(names_data)
            and sorted(names_expected) == sorted(names_data)
        ):
            raise ValueError(
                "If data is passed as a DataFrame, the"
                "column names must match the channel names in `nm_channels`.\n"
                f"Input dataframe column names: {names_data}\n"
                f'Expected (from nm_channels["name"]): : {names_expected}.'
            )
        return data.to_numpy().transpose()

    def _run(
        self,
        data: np.ndarray | pd.DataFrame | None = None,
        out_path_root: _PathLike = "",
        folder_name: str = "sub",
        is_stream_lsl: bool = True,
        stream_lsl_name: str = None,
        plot_lsl: bool = False,
        save_csv: bool = False,
        save_interval: int = 10,
    ) -> pd.DataFrame:
        from py_neuromodulation.nm_database import NMDatabase

        out_path_root = Path.cwd() if not out_path_root else Path(out_path_root)
        out_dir = out_path_root / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        db = NMDatabase(out_dir, folder_name)  # Create output database

        self.batch_count: int = 0  # Keep track of the number of batches processed

        if not is_stream_lsl:
            from py_neuromodulation.nm_generator import raw_data_generator

            generator = raw_data_generator(
                data=data,
                settings=self.settings,
                sfreq=self.sfreq,
            )
        else:
            from py_neuromodulation.nm_mnelsl_stream import LSLStream

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
                logger.warning(error_msg)
                self.sfreq = self.lsl_stream.stream.sinfo.sfreq

            generator = self.lsl_stream.get_next_batch()

        prev_batch_end = 0

        while True:
            next_item = next(generator, None)

            if next_item is not None:
                timestamps, data_batch = next_item
            else:
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
                batch_length if is_stream_lsl else np.ceil(this_batch_end * 1000 + 1)
            )

            prev_batch_end = this_batch_end

            if self.verbose:
                logger.info("Time: %.2f", feature_dict["time"] / 1000)

            self._add_target(feature_dict, data_batch)

            # We should ensure that feature output is float64 and remove this
            for key, value in feature_dict.items():
                feature_dict[key] = float(value)

            db.insert_data(feature_dict)

            self.batch_count += 1
            if self.batch_count % save_interval == 0:
                db.commit()

        # Save last batches and close the database connection
        db.commit()
        db.close()

        # If save_csv is False, still save the first row to get the column names
        feature_df = db.fetch_all() if save_csv else db.head()

        self.save_after_stream(out_dir, feature_df)

        return feature_df  # Not sure if this makes sense anymore

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

        if self.nm_channels is not None:
            ch_names = self.nm_channels["name"].to_list()
            ch_types = self.nm_channels["type"].to_list()
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


class Stream(_GenericStream):
    def __init__(
        self,
        sfreq: float,
        data: np.ndarray | pd.DataFrame | None = None,
        nm_channels: pd.DataFrame | _PathLike | None = None,
        settings: "NMSettings | _PathLike | None" = None,
        sampling_rate_features_hz: float | None = None,
        line_noise: float | None = 50,
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
        data : np.ndarray | pd.DataFrame | None, optional
            data to be streamed with shape (n_channels, n_time), by default None
        nm_channels : pd.DataFrame | _PathLike
            parametrization of channels (see nm_define_channels.py for initialization)
        settings : NMSettings | _PathLike | None, optional
            Initialized nm_settings.NMSettings object, by default the py_neuromodulation/nm_settings.yaml are read
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
            log stream computation time information, by default True
        """

        if nm_channels is None and data is not None:
            from py_neuromodulation.nm_define_nmchannels import (
                get_default_channels_from_data,
            )

            nm_channels = get_default_channels_from_data(data)

        if nm_channels is None and data is None:
            raise ValueError(
                "Either `nm_channels` or `data` must be passed to `Stream`."
            )

        super().__init__(
            sfreq=sfreq,
            nm_channels=nm_channels,
            settings=settings,
            line_noise=line_noise,
            sampling_rate_features_hz=sampling_rate_features_hz,
            path_grids=path_grids,
            coord_names=coord_names,
            coord_list=coord_list,
            verbose=verbose,
        )

        self.data = data

        self.target_idx_initialized: bool = False

    def run(
        self,
        data: np.ndarray | pd.DataFrame | None = None,
        out_path_root: _PathLike = Path.cwd(),
        folder_name: str = "sub",
        stream_lsl: bool = False,
        stream_lsl_name: str = None,
        save_csv: bool = False,
        plot_lsl: bool = False,
        save_interval: float = 1.0,
    ) -> pd.DataFrame:
        """Call run function for offline stream.

        Parameters
        ----------
        data : np.ndarray | pd.DataFrame
            shape (n_channels, n_time)
        out_path_root : _PathLike | None, optional
            Full path to store estimated features, by default None
            If None, data is simply returned and not saved
        folder_name : str, optional
             folder output name, commonly subject or run name, by default "sub"
        stream_lsl : bool, optional
            stream data from LSL, by default False
        stream_lsl_name : str, optional
            stream name, by default None
        plot_lsl : bool, optional
            plot data with mne_lsl stream_viewer
        save_csv : bool, optional
            save csv file, by default False
        save_interval : int, optional
            save interval in number of samples, by default 10

        Returns
        -------
        pd.DataFrame
            feature DataFrame
        """

        super().run()  # reinitialize the stream

        self.stream_lsl = stream_lsl
        self.stream_lsl_name = stream_lsl_name

        if data is not None:
            data = self._handle_data(data)
        elif self.data is not None:
            data = self._handle_data(self.data)
        elif self.data is None and data is None and self.stream_lsl is False:
            raise ValueError("No data passed to run function.")

        out_path = Path(out_path_root, folder_name)
        out_path.mkdir(parents=True, exist_ok=True)
        logger.log_to_file(out_path)

        return self._run(
            data,
            out_path_root,
            folder_name,
            is_stream_lsl=stream_lsl,
            stream_lsl_name=stream_lsl_name,
            save_csv=save_csv,
            plot_lsl=plot_lsl,
            save_interval=save_interval,
        )
