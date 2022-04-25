import os

import numpy as np
import pandas as pd

import py_neuromodulation as py_nm


class EpochStream(py_nm.nm_stream.PNStream):
    def __init__(self) -> None:
        super().__init__()

    def read_epoch_data(self, path_epoch) -> None:
        """Read npy array of epochs. Shape is assumed to be (samples, channels, time)

        Parameters
        ----------
        path_epoch : str
        """
        self.data = np.load(path_epoch)

    def get_data_gen_all_data(
        self,
    ) -> np.array:
        """This data generator returns one epoch at a time.
        Data will thus be analyzed in steps of the epoch size

        Returns
        -------
        np.array
            _description_

        Yields
        ------
        Iterator[np.array]
            _description_
        """
        for n_batch in range(self.data.shape[0]):
            yield self.data[n_batch, :, :]

    def get_data_gen_epoch(self) -> np.array:
        self.settings["sampling_rate_features_hz"]

    def run(
        self,
    ):

        self._set_run()
        # shape is n, channels=7, 800 Hz

        self.feature_arr = pd.DataFrame()
        data_gen = self.get_data()
        idx = 0

        while True:
            data = next(
                data_gen, None
            )  # None will be returned if generator ran through
            if data is None:
                break

            feature_series = self.run_analysis.process_data(data)
            if idx == 0:
                self.feature_arr = pd.DataFrame([feature_series])
                idx += 1
            else:
                self.feature_arr = self.feature_arr.append(
                    feature_series, ignore_index=True
                )

        self.feature_arr.to_csv("features_epoch.csv")

    def _add_timestamp(self, feature_series: pd.Series, idx: int = None) -> pd.Series:
        # in case of epochs no timestamp is necessary
        return feature_series

    def _add_coordinates(self) -> None:
        pass
