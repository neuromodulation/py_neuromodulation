import multiprocessing
import os
import sys
import pathlib
from typing import MutableMapping
import numpy as np
import pandas as pd
import time
import timeit
from pynput.keyboard import Key, Listener

from py_neuromodulation import \
    (nm_projection,
    nm_rereference,
    nm_run_analysis,
    nm_features,
    nm_resample,
    nm_stream,
    nm_test_settings,
    FieldTrip)


class RealTimePyNeuro(nm_stream.PNStream):

    queue_raw:multiprocessing.Queue = multiprocessing.Queue(1)
    queue_features:multiprocessing.Queue = multiprocessing.Queue(1)
    ftc:FieldTrip.Client = None
    filename: str

    def __init__(
        self,
        PATH_SETTINGS: str,
        PATH_NM_CHANNELS: str,
        PATH_OUT: str = os.getcwd(),
        filename: str = "testrun",
        PATH_GRIDS: str = pathlib.Path(__file__).parent.resolve(),
        VERBOSE: bool = True,
        fs: int = 128,
        line_noise: int = 50
    ) -> None:

        super().__init__(PATH_SETTINGS=PATH_SETTINGS,
            PATH_NM_CHANNELS=PATH_NM_CHANNELS,
            PATH_OUT=PATH_OUT,
            PATH_GRIDS=PATH_GRIDS,
            VERBOSE=VERBOSE)

        self.set_fs(fs)
        self.set_linenoise(line_noise)
        self.filename = filename

        self.nm_channels = self._get_nm_channels(
            self.PATH_NM_CHANNELS
        )

        # leave out coordinate setting for now

        self._set_run()

        self.ftc = self.init_fieldtrip()

    @staticmethod
    def init_fieldtrip() -> FieldTrip.Client:
        """Initialize Fieldtrip client

        Returns
        -------
        FieldTrip.Client
        """
        ftc = FieldTrip.Client()
        # Python FieldTripBuffer https://www.fieldtriptoolbox.org/development/realtime/buffer_python/
        ftc.connect('localhost', 1972)   # might throw IOError
        H = ftc.getHeader()

        if H is None:
            print('Failed to retrieve header!')
            sys.exit(1)

        print(H)
        print(H.labels)

        return ftc

    def run(self) -> None:
        """Start get_data, calcFeatures and sendFeature processes
        """
        processes = [
            multiprocessing.Process(
                target=self.get_data,
                args=(
                    self.queue_raw,
                    self.ftc,
                )
            ),
            multiprocessing.Process(
                target=self.calcFeatures, args=(
                    self.queue_raw,
                    self.queue_features,
                )
            ),
            multiprocessing.Process(
                target=self.sendFeatures,
                args=(
                    self.queue_features,
                    self.ftc,
                )
            )
        ]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    def get_data(
        self,
        queue_raw: multiprocessing.Queue,
        ftc: FieldTrip.Client
    ) -> np.array:

        def on_press(key):
            print('{0} pressed'.format(
                key))

        def on_release(key):
            if key == Key.esc:
                # Stop listener
                print("reiceived stop key pressed")
                queue_raw.put(None)
                return False

        listener = Listener(
            on_press=on_press,
            on_release=on_release)
        listener.start()
        last_batch = 0

        while listener.is_alive() is True:
            # read new data
            print('Trying to read last sample...')
            #index = H.nSamples - 1
            ieeg_batch = ftc.getData()

            #number_repeat = 10
            #val = timeit.timeit(
            #    lambda: ftc.getData(),
            #    number=number_repeat
            #) / number_repeat

            ieeg_batch = ieeg_batch[-128:, :2].T  # take last, #1: data
            # check if time stamp changed 
            if np.array_equal(ieeg_batch, last_batch) is False:
                last_batch = ieeg_batch
                queue_raw.put(ieeg_batch)
            else:
                queue_raw.put(ieeg_batch)
        ftc.disconnect()

    def calcFeatures(
        self,
        queue_raw: multiprocessing.Queue,
        queue_features: multiprocessing.Queue
    ):

        FLAG_STOP = False
        while FLAG_STOP is False:
            ieeg_batch = queue_raw.get()  # ch, samples
            if ieeg_batch is None:
                queue_features.put(None)
                FLAG_STOP = True
            else:
                feature_series = \
                    self.run_analysis.process_data(ieeg_batch)
                feature_series = self._add_timestamp(feature_series)
                print("calc features")
                queue_features.put(feature_series)

    def sendFeatures(
        self,
        queue_features: multiprocessing.Queue,
        ftc: FieldTrip.Client
    ):

        idx = 0
        IDENT_FEATURES = 21
        FLAG_STOP = False
        while FLAG_STOP is False:
            features = queue_features.get()

            if features is None:
                self.save_after_stream(self.filename)
                print("SAVED")
                FLAG_STOP = True
            else:
                if idx == 0:
                    self.feature_arr = pd.DataFrame([features])
                    idx += 1
                else:
                    self.feature_arr = self.feature_arr.append(
                        features,
                        ignore_index=True
                    )

                print("length of features:" + str(len(self.feature_arr)))
                H = ftc.getHeader()
                # channel names are 1. data 2. ident 3. timestamp
                # put at the end of the buffer the calculated features of the data channel
                to_send = np.zeros([H.nSamples, H.nChannels])
                #to_send[-features.shape[0]:, IDENT_FEATURES] = np.array(features)
                to_send[-features.shape[0]:, 1] = np.array(features)
                ftc.putData(to_send)

                #number_repeat = 10
                #val = timeit.timeit(
                #    lambda: ftc.putData(to_send),
                #    number=number_repeat
                #) / number_repeat

    def _add_timestamp(self, feature_series: pd.Series, idx: int = None) -> pd.Series:

        feature_series["time"] = time.time()  # UNIX Timestamp

        return feature_series

    def _add_coordinates(self):
        """Lateron add here method for providing coordinates"""
        pass
