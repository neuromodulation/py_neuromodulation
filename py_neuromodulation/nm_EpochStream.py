import os
import pickle

import numpy as np
import pandas as pd

import py_neuromodulation as py_nm
from py_neuromodulation import nm_generator


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

    def get_data(
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

    def run(
        self,
    ):

        self._set_run()
        # shape is n, channels=7, 800 Hz

        self.feature_arr = pd.DataFrame()
        self.feature_arr_list = []
        epoch_gen = self.get_data()
        idx_epoch = 0

        while True:
            data = next(
                epoch_gen, None
            )  # None will be returned if generator ran through
            if data is None:
                break
            gen = nm_generator.ieeg_raw_generator(data, self.settings, self.fs)

            def get_data_within_epoch() -> np.array:
                return next(gen, None)

            idx_within_epoch = 0
            while True:
                data_within_epoch = get_data_within_epoch()
                if data_within_epoch is None:
                    break

                feature_series = self.run_analysis.process_data(data_within_epoch)
                if idx_within_epoch == 0:
                    self.feature_arr = pd.DataFrame([feature_series])
                    idx_within_epoch += 1
                else:
                    self.feature_arr = self.feature_arr.append(
                        feature_series, ignore_index=True
                    )
            self.feature_arr_list.append(self.feature_arr)

    def _add_timestamp(self, feature_series: pd.Series, idx: int = None) -> pd.Series:
        # in case of epochs no timestamp is necessary
        return feature_series

    def _add_coordinates(self) -> None:
        pass
