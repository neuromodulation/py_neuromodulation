from py_neuromodulation.utils import logger
from py_neuromodulation.utils.io import MNE_FORMATS, read_mne_data, load_channels
from py_neuromodulation.utils.types import _PathLike
from py_neuromodulation.utils import create_channels
from .data_generator_abc import DataGeneratorABC
import numpy as np
import pandas as pd
from typing import Tuple

class RawDataGenerator(DataGeneratorABC):
    """
    This generator function mimics online data acquisition.
    The data are iteratively sampled with settings.sampling_rate_features_hz
    """

    def __init__(
        self,
        data: "np.ndarray | pd.DataFrame | _PathLike | None",
        sampling_rate_features_hz: float,
        segment_length_features_ms: float,
        channels: "pd.DataFrame | None",
        sfreq: "float | None",
    ) -> None:
        """
        Arguments
        ---------
            data (np array): shape (channels, time)
            settings (settings.NMSettings): settings object
            sfreq (float): sampling frequency of the data

        Returns
        -------
            np.array: 1D array of time stamps
            np.array: new batch for run function of full segment length shape
        """
        self.channels = channels
        self.sfreq = sfreq
        self.batch_counter: int = 0  # counter for the batches
        self.target_idx_initialized: bool = False

        if isinstance(data, (np.ndarray, pd.DataFrame)):
            logger.info(f"Loading data from {type(data).__name__}")
            self.data = data
        elif isinstance(self.data, _PathLike):
            logger.info("Loading data from file")
            filepath = Path(self.data)  # type: ignore
            ext = filepath.suffix

            if ext in MNE_FORMATS:
                data, sfreq, ch_names, ch_types, bads = read_mne_data(filepath)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            self.channels = create_channels(
                ch_names=ch_names,
                ch_types=ch_types,
                used_types=["eeg", "ecog", "dbs", "seeg"],
                bads=bads,
            )
            
            if sfreq is None:
                raise ValueError(
                    "Sampling frequency not specified in file, please specify sfreq as a parameters"
                )
            self.sfreq = sfreq
            self.data = self._handle_data(data)
        else:
            raise ValueError(
                "Data must be either a numpy array, a pandas DataFrame, or a path to an MNE supported file"
            )
        self.sfreq = sfreq
        # Width, in data points, of the moving window used to calculate features
        self.segment_length = segment_length_features_ms / 1000 * sfreq
        # Ratio of the sampling frequency of the input data to the sampling frequency
        self.stride = sfreq / sampling_rate_features_hz

        self.channels = load_channels(channels) if channels is not None else None
    
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

    def add_target(self, feature_dict: "pd.DataFrame", data_batch: np.array) -> None:
        """Add target channels to feature series.

        Parameters
        ----------
        feature_dict : pd.DataFra,e

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
                feature_dict[target_name] = data_batch[target_idx, -1]

        return feature_dict

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        start = self.stride * self.batch_counter
        end = start + self.segment_length

        self.batch_counter += 1

        start_idx = int(start)
        end_idx = int(end)

        if end_idx > self.data.shape[1]:
            raise StopIteration

        return np.arange(start, end) / self.sfreq, self.data[:, start_idx:end_idx]
