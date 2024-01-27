"""Module for offline data streams."""
import os
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from itertools import count

import mne

from py_neuromodulation import (
    nm_generator,
    nm_IO,
    nm_stream_abc,
    nm_define_nmchannels,
)

_PathLike = str | os.PathLike


class _OfflineStream(nm_stream_abc.PNStream):
    """Offline stream base class.
    This class can be inhereted for different types of offline streams, e.g. epoch-based or continuous.

    Parameters
    ----------
    nm_stream_abc : nm_stream_abc.PNStream
    """

    def _add_target(
        self, feature_series: pd.Series, data: np.ndarray
    ) -> pd.Series:
        """Add target channels to feature series.

        Parameters
        ----------
        feature_series : pd.Series
        data : np.ndarray
            Raw data with shape (n_channels, n_samples). Channels not for feature computation are also included

        Returns
        -------
        pd.Series
            feature series with target channels added
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

            for target_idx, target_name in zip(
                self.target_indexes, self.target_names
            ):
                feature_series[target_name] = data[target_idx, -1]
        return feature_series

    def _add_timestamp(
        self, feature_series: pd.Series, cnt_samples: int
    ) -> pd.Series:
        """Add time stamp in ms.

        Due to normalization run_analysis needs to keep track of the counted
        samples. These are accessed here for time conversion.
        """
        feature_series["time"] = cnt_samples * 1000 / self.sfreq

        if self.verbose:
            print(
                str(np.round(feature_series["time"] / 1000, 2))
                + " seconds of data processed"
            )

        return feature_series

    def _handle_data(self, data: np.ndarray | pd.DataFrame) -> np.ndarray:
        names_expected = self.nm_channels["name"].to_list()

        if isinstance(data, np.ndarray):
            if not len(names_expected) == data.shape[0]:
                raise ValueError(
                    "If data is passed as an array, the first dimension must"
                    " match the number of channel names in `nm_channels`. Got:"
                    f" Data columns: {data.shape[0]}, nm_channels.name:"
                    f" {len(names_expected)}."
                )
            return data
        names_data = data.columns.to_list()
        if not (
            len(names_expected) == len(names_data)
            and sorted(names_expected) == sorted(names_data)
        ):
            raise ValueError(
                "If data is passed as a DataFrame, the"
                "columns must match the channel names in `nm_channels`. Got:"
                f"Data columns: {names_data}, nm_channels.name: {names_data}."
            )
        return data.to_numpy()
    
    def _check_settings_for_parallel(self):
        """Check specified settings and raise error if parallel processing is not possible.

        Raises:
            ValueError: depending on the settings, parallel processing is not possible
        """

        if "raw_normalization" in self.settings["preprocessing"]:
            raise ValueError(
                "Parallel processing is not possible with raw_normalization normalization."
            )
        if self.settings["postprocessing"]["feature_normalization"] is True:
            raise ValueError(
                "Parallel processing is not possible with feature normalization."
            )
        if self.settings["features"]["bursts"] is True:
            raise ValueError(
                "Parallel processing is not possible with burst estimation."
            )


    def _process_batch(self, data_batch, cnt_samples):
        feature_series = self.run_analysis.process(
            data_batch.astype(np.float64)
        )
        feature_series = self._add_timestamp(feature_series, cnt_samples)
        return feature_series
        
    def _run_offline(
        self,
        data: np.ndarray,
        out_path_root: _PathLike | None = None,
        folder_name: str = "sub",
        parallel: bool = False,
        n_jobs: int = -2,
    ) -> pd.DataFrame:
        generator = nm_generator.raw_data_generator(
            data=data,
            settings=self.settings,
            sfreq=self.sfreq,
        )

        sample_add = self.sfreq / self.run_analysis.sfreq_features

        offset_time = self.settings["segment_length_features_ms"]
        # offset_start = np.ceil(offset_time / 1000 * self.sfreq).astype(int)
        offset_start = offset_time / 1000 * self.sfreq

        if parallel:
            l_features = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(self._process_batch)(data_batch, cnt_samples)
                for data_batch, cnt_samples in zip(
                    generator, count(offset_start, sample_add)
                )
            )

        else:
            l_features = []
            cnt_samples = offset_start
            while True:
                data_batch = next(generator, None)
                if data_batch is None:
                    break
                feature_series = self.run_analysis.process(
                    data_batch.astype(np.float64)
                )
                feature_series = self._add_timestamp(
                    feature_series, cnt_samples
                )
                l_features.append(feature_series)

                cnt_samples += sample_add
        feature_df = pd.DataFrame(l_features)

        feature_df = self._add_target(feature_series=feature_df, data=data)

        self.save_after_stream(out_path_root, folder_name, feature_df)

        return feature_df

    def plot_raw_signal(
        self,
        sfreq: float = None,
        data: np.array = None,
        lowpass: float = None,
        highpass: float = None,
        picks: list = None,
        plot_time: bool = True,
        plot_psd: bool = False,
    ) -> None:
        """Use MNE-RawArray Plot to investigate PSD or raw_signal plot.

        Parameters
        ----------
        sfreq : float
            sampling frequency [Hz]
        data : np.array, optional
            data (n_channels, n_times), by default None
        plot_time : bool, optional
            mne.io.RawArray.plot(), by default True
        plot_psd : bool, optional
            mne.io.RawArray.plot(), by default True

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

        # create mne.RawArray
        info = mne.create_info(
            ch_names=ch_names, sfreq=sfreq, ch_types=ch_types
        )
        raw = mne.io.RawArray(data, info)

        if picks is not None:
            raw = raw.pick(picks)
        self.raw = raw
        if plot_time:
            raw.plot(highpass=highpass, lowpass=lowpass)
        if plot_psd:
            raw.compute_psd().plot()


class Stream(_OfflineStream):
    def __init__(
        self,
        sfreq: int | float,
        data: np.ndarray | pd.DataFrame = None,
        nm_channels: pd.DataFrame | _PathLike = None,
        settings: dict | _PathLike | None = None,
        sampling_rate_features_hz: float = None,
        line_noise: int | float | None = 50,
        path_grids: _PathLike | None = None,
        coord_names: list | None = None,
        coord_list: list | None = None,
        verbose: bool = True,
    ) -> None:
        """Stream initialization

        Parameters
        ----------
        sfreq : int | float
            sampling frequency of data in Hertz
        data : np.ndarray | pd.DataFrame | None, optional
            data to be streamed with shape (n_channels, n_time), by default None
        nm_channels : pd.DataFrame | _PathLike
            parametrization of channels (see nm_define_channels.py for initialization)
        settings : dict | _PathLike | None, optional
            features settings can be a dictionary or path to the nm_settings.json, by default the py_neuromodulation/nm_settings.json are read
        line_noise : int | float | None, optional
            line noise, by default 50
        sampling_rate_features_hz : int | float | None, optional
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

        if nm_channels is None and data is not None:
            nm_channels = nm_define_nmchannels.get_default_channels_from_data(
                data
            )

        if nm_channels is None and data is None:
            raise ValueError(
                "Either `nm_channels` or `data` must be passed to `Stream`."
            )

        super().__init__(
            sfreq,
            nm_channels,
            settings,
            line_noise,
            sampling_rate_features_hz,
            path_grids,
            coord_names,
            coord_list,
            verbose,
        )

        self.data = data

        self.target_idx_initialized = False

    def run(
        self,
        data: np.ndarray | pd.DataFrame = None,
        out_path_root: _PathLike | None = None,
        folder_name: str = "sub",
        parallel: bool = False,
        n_jobs: int = -2
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

        Returns
        -------
        pd.DataFrame
            feature DataFrame
        """

        super().run()  # reinitialize the stream

        if data is not None:
            data = self._handle_data(data)
        elif self.data is not None:
            data = self._handle_data(self.data)
        elif self.data is None and data is None:
            raise ValueError("No data passed to run function.")
        
        if parallel is True:
            self._check_settings_for_parallel()

        return self._run_offline(data, out_path_root, folder_name, parallel=parallel, n_jobs=n_jobs)
