from typing import Optional, Iterable, Union
import os
import pathlib
import mne
import numpy as np
import pandas as pd
import timeit

from py_neuromodulation import nm_IO, nm_generator, nm_stream


class GenericStream(nm_stream.PNStream):

    PATH_ANNOTATIONS: str
    PATH_GRIDS: str
    PATH_DATA: str
    PATH_RUN: str
    LIMIT_DATA: bool
    LIMIT_LOW: int
    LIMIT_HIGH: int
    used_types: bool
    raw_arr: mne.io.RawArray
    raw_arr_data: np.array
    annot: mne.Annotations
    VERBOSE: bool = False
    READ_BIDS: bool = False

    def __init__(
        self,
        PATH_RUN: str = None,
        PATH_SETTINGS: str = os.path.join(
            pathlib.Path(__file__).parent.resolve(), "nm_settings.json"
        ),
        PATH_NM_CHANNELS: str = str(),
        PATH_OUT: str = os.getcwd(),
        PATH_GRIDS: str = pathlib.Path(__file__).parent.resolve(),
        VERBOSE: bool = True,
        PATH_ANNOTATIONS: str = str(),
        PATH_BIDS: str = str(),
        LIMIT_DATA: bool = False,
        LIMIT_LOW: int = 0,
        LIMIT_HIGH: int = 10000,
        READ_BIDS: bool = False,
        ch_names: Optional[Iterable[str]] = None,
        ch_types: Optional[Iterable[str]] = None,
        bads: Optional[Iterable[int]] = None,
        fs: float = None,
        line_noise: float = None,
        coord_names: Optional[Iterable[str]] = None,
        coord_list: Optional[Iterable[str]] = None,
        reference: Optional[Union[str, Iterable[str]]] = "default",
        used_types: Optional[Iterable[str]] = ("ecog", "dbs", "seeg"),
        target_keywords: Optional[Iterable[str]] = ("mov", "squared", "label"),
    ) -> None:

        super().__init__(
            PATH_SETTINGS=PATH_SETTINGS,
            PATH_NM_CHANNELS=PATH_NM_CHANNELS,
            PATH_OUT=PATH_OUT,
            PATH_GRIDS=PATH_GRIDS,
            VERBOSE=VERBOSE,
            ch_names=ch_names,
            ch_types=ch_types,
            bads=bads,
            fs=fs,
            line_noise=line_noise,
            reference=reference,
        )

        self.PATH_RUN = PATH_RUN
        self.PATH_ANNOTATIONS = PATH_ANNOTATIONS
        self.PATH_BIDS = PATH_BIDS
        self.used_types = used_types
        self.LIMIT_DATA = LIMIT_DATA
        self.LIMIT_LOW = LIMIT_LOW
        self.LIMIT_HIGH = LIMIT_HIGH
        self.coord_names = coord_names
        self.coord_list = coord_list

        if READ_BIDS is True:
            self.raw_arr, raw_arr_data, fs, line_noise = nm_IO.read_BIDS_data(
                self.PATH_RUN, self.PATH_DATA
            )
            self.set_data(raw_arr_data)
            self.ch_types = self.raw_arr.get_channel_types()
            self.ch_names = self.raw_arr.ch_names
            self.bads = self.raw_arr.info["bads"]

            self.coord_list, self.coord_names = self._get_bids_coord_list(self.raw_arr)

            # read BIDS annotations
            if self.PATH_ANNOTATIONS:
                self.annot, self.annot_data, self.raw_arr = nm_IO.get_annotations(
                    self.PATH_ANNOTATIONS, self.PATH_RUN, self.raw_arr
                )

        self.nm_channels = self._get_nm_channels(
            PATH_NM_CHANNELS,
            ch_names=self.ch_names,
            ch_types=self.ch_types,
            bads=self.bads,
            used_types=used_types,
            target_keywords=target_keywords,
            reference=self.reference,
        )

        if self.coord_list is None and True in [
            self.settings["methods"]["project_cortex"],
            self.settings["methods"]["project_subcortex"],
        ]:
            raise ValueError("no coordinates could be loaded")
        elif self.coord_list is not None:
            self.coords = self._add_coordinates(self.coord_names, self.coord_list)
            self.sess_right = self._get_sess_lat(self.coords)

    def set_data(self, data: np.array):
        """Set Data should be called after init without BIDS data.

        Parameters
        ----------
        data : np.array
            shape (channels, time)
        """
        self.raw_arr_data = data

        if self.LIMIT_DATA:
            self.raw_arr_data = self.raw_arr_data[:, self.LIMIT_LOW : self.LIMIT_HIGH]

        self.gen = nm_generator.ieeg_raw_generator(
            self.raw_arr_data, self.settings, self.fs
        )

    def get_data(self) -> np.array:
        return next(self.gen, None)

    def run(self, predict: bool = False) -> None:
        """BIDS specific fun function
        Does not need to run in parallel
        """
        # init features, projection etc. here
        # settings, nm_channels might have been changed by the user in e.g. the ipynb
        self._set_run()

        idx = 0
        while True:
            data = self.get_data()
            if data is None:
                break
            feature_series = self.run_analysis.process_data(data)

            # Measuring timing
            # number_repeat = 100
            # val = timeit.timeit(
            #    lambda: self.run_analysis.process_data(data),
            #    number=number_repeat
            # ) / number_repeat

            feature_series = self._add_timestamp(feature_series, idx)

            # concatenate data to feature_arr
            if idx == 0:
                self.feature_arr = pd.DataFrame([feature_series])
                idx += 1
            else:
                self.feature_arr = pd.concat([
                    self.feature_arr, feature_series.to_frame().T
                ], ignore_index=False, axis=0)

            if predict is True:
                prediction = self.model.predict(feature_series)

        self._add_labels()

        if self.READ_BIDS:
            folder_name = os.path.basename(self.PATH_RUN)
            self.save_after_stream(folder_name)

    def _add_timestamp(self, feature_series: pd.Series, idx: int = None) -> pd.Series:
        """time stamp is added in ms
        Due to normalization run_analysis needs to keep track of the counted samples
        Those are accessed here for time conversion"""

        if idx == 0:
            feature_series["time"] = self.run_analysis.offset
        else:
            # sampling frequency is taken here from run_analysis, since resampling might change it
            feature_series["time"] = (
                self.run_analysis.cnt_samples * 1000 / self.run_analysis.fs
            )

        if self.VERBOSE:
            print(
                str(np.round(feature_series["time"] / 1000, 2))
                + " seconds of data processed"
            )

        return feature_series

    def _add_labels(self):
        """add resampled labels to feature dataframe if there are target channels"""
        if self.nm_channels.target.sum() > 0:
            self.feature_arr = nm_IO.add_labels(
                self.feature_arr,
                self.settings,
                self.nm_channels,
                self.raw_arr_data,
                self.fs,
            )
        else:
            pass

    def _get_bids_coord_list(self, raw_arr: mne.io.RawArray):
        if raw_arr.get_montage() is not None:
            coord_list = np.array(
                list(dict(raw_arr.get_montage().get_positions()["ch_pos"]).values())
            ).tolist()
            coord_names = np.array(
                list(dict(raw_arr.get_montage().get_positions()["ch_pos"]).keys())
            ).tolist()
        else:
            coord_list = None
            coord_names = None

        return coord_list, coord_names

    def _add_coordinates(self, coord_names: list, coord_list: np.array):
        """set coordinate information to settings from RawArray
        The set coordinate positions are set as lists,
        since np.arrays cannot be saved in json
        Parameters
        ----------
        raw_arr : mne.io.RawArray
        PATH_GRIDS : string, optional
            absolute path to grid_cortex.tsv and grid_subcortex.tsv, by default: None
        """

        coords = {}

        def left_coord(val, coord_region):
            if coord_region.split("_")[1] == "left":
                return val < 0
            else:
                return val > 0

        for coord_region in [
            coord_loc + "_" + lat
            for coord_loc in ["cortex", "subcortex"]
            for lat in ["left", "right"]
        ]:

            coords[coord_region] = {}

            ch_type = "ECOG" if "cortex" == coord_region.split("_")[0] else "LFP"

            coords[coord_region]["ch_names"] = [
                coord_names[ch_idx]
                for ch_idx, ch in enumerate(coord_list)
                if left_coord(coord_list[ch_idx][0], coord_region)
                and (ch_type in coord_names[ch_idx])
            ]

            # multiply by 1000 to get m instead of mm
            coords[coord_region]["positions"] = 1000 * np.array(
                [
                    ch
                    for ch_idx, ch in enumerate(coord_list)
                    if left_coord(coord_list[ch_idx][0], coord_region)
                    and (ch_type in coord_names[ch_idx])
                ]
            )

        return coords
