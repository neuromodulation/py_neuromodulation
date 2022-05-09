import os
import timeit

import mne
import mne_bids
import numpy as np
import pandas as pd

from py_neuromodulation import nm_generator, nm_IO, nm_stream_abc

_PathLike = str | os.PathLike


class _OfflineStream(nm_stream_abc.PNStream):
    def _run_offline(
        self,
        data: np.ndarray,
    ) -> None:
        self._set_run()

        generator = nm_generator.ieeg_raw_generator(
            data, self.settings, self.sfreq
        )
        features = []
        first_sample = True
        while True:
            data_batch = next(generator, None)
            if data_batch is None:
                break
            feature_series = self.run_analysis.process_data(data_batch)

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

        feature_df = pd.DataFrame(features)
        feature_df = self._add_labels(features=feature_df, data=data)

        self.save_after_stream(
            os.path.basename(str(self.path_out)), feature_df
        )

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

    def _add_timestamp(
        self, feature_series: pd.Series, first_sample: bool
    ) -> pd.Series:
        """time stamp is added in ms
        Due to normalization run_analysis needs to keep track of the counted samples
        Those are accessed here for time conversion"""

        if first_sample:
            feature_series["time"] = self.run_analysis.offset
        else:
            # sampling frequency is taken from run_analysis, since resampling
            # might change it
            feature_series["time"] = (
                self.run_analysis.cnt_samples * 1000 / self.run_analysis.fs
            )

        if self.verbose:
            print(
                str(np.round(feature_series["time"] / 1000, 2))
                + " seconds of data processed"
            )

        return feature_series

    def _add_labels(
        self, features: pd.DataFrame, data: np.ndarray
    ) -> pd.DataFrame:
        """Add resampled labels to features if there are target channels."""
        if self.nm_channels.target.sum() > 0:
            features = nm_IO.add_labels(
                df_=features,
                settings=self.settings,
                nm_channels=self.nm_channels,
                raw_arr_data=data,
                fs=self.sfreq,
            )
        return features


class Stream(_OfflineStream):
    def run(
        self,
        data: np.ndarray | pd.DataFrame,
        sfreq: int | float,
        coord_names: list | None = None,
        coord_list: list | None = None,
    ) -> None:
        """BIDS specific fun function. Does not need to run in parallel."""
        self.sfreq = int(sfreq)

        data = self._handle_data(data)
        self.coords = self._set_coords(
            coord_names=coord_names, coord_list=coord_list
        )
        self._set_run()

        self._run_offline(data)


class BidsStream(_OfflineStream):
    def run(
        self,
        filepath: _PathLike | mne_bids.BIDSPath,
        datatype: str,
        bids_root: _PathLike | None = None,
    ) -> None:
        """Run stream with BIDS file."""
        raw, data, sfreq, _ = nm_IO.read_BIDS_data(
            PATH_RUN=filepath,
            BIDS_PATH=bids_root,
            datatype=datatype,
        )
        coord_list, coord_names = _get_coord_list(raw)
        self.coords = self._set_coords(
            coord_names=coord_names, coord_list=coord_list
        )
        self.sfreq = int(sfreq)

        self._run_offline(data)


def _get_coord_list(
    raw: mne.io.BaseRaw,
) -> tuple[list, list] | tuple[None, None]:
    montage = raw.get_montage()
    if montage is not None:
        coord_list = np.array(
            list(dict(montage.get_positions()["ch_pos"]).values())
        ).tolist()
        coord_names = np.array(
            list(dict(montage.get_positions()["ch_pos"]).keys())
        ).tolist()
    else:
        coord_list = None
        coord_names = None

    return coord_list, coord_names
