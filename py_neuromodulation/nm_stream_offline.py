"""Module for offline data streams."""
import os

# import timeit

import numpy as np
import pandas as pd

from py_neuromodulation import nm_generator, nm_IO, nm_stream_abc

_PathLike = str | os.PathLike


class _OfflineStream(nm_stream_abc.PNStream):
    def _add_labels(
        self, features: pd.DataFrame, data: np.ndarray
    ) -> pd.DataFrame:
        """Add resampled labels to features if there are target channels."""
        if self.nm_channels["target"].sum() > 0:
            features = nm_IO.add_labels(
                features=features,
                settings=self.settings,
                nm_channels=self.nm_channels,
                raw_arr_data=data,
                fs=self.sfreq,
            )
        return features

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

    def _run_offline(
        self,
        data: np.ndarray,
        out_path_root: _PathLike | None = None,
        folder_name: str = "sub",
    ) -> None:
        generator = nm_generator.raw_data_generator(
            data=data,
            settings=self.settings,
            sfreq=self.sfreq,
        )
        features = []
        first_sample = True
        sample_add = int(self.sfreq / self.run_analysis.sfreq_features)
        cnt_samples = int(self.sfreq)

        while True:
            data_batch = next(generator, None)
            if data_batch is None:
                break
            feature_series = self.run_analysis.process(data_batch)

            # Measuring timing
            # number_repeat = 100
            # val = timeit.timeit(
            #    lambda: self.run_analysis.process_data(data),
            #    number=number_repeat
            # ) / number_repeat

            feature_series = self._add_timestamp(feature_series, first_sample)
            features.append(feature_series)

            if self.model is not None:
                prediction = self.model.predict(feature_series)

            if first_sample:
                first_sample = False

            cnt_samples += sample_add

        feature_df = pd.DataFrame(features)
        feature_df = self._add_labels(features=feature_df, data=data)

        self.save_after_stream(out_path_root, folder_name, feature_df)


class Stream(_OfflineStream):
    def run(
        self,
        data: np.ndarray | pd.DataFrame,
        out_path_root: _PathLike | None = None,
        folder_name: str = "sub",
    ) -> None:
        """Does not need to run in parallel."""

        data = self._handle_data(data)

        self._run_offline(data, out_path_root, folder_name)
