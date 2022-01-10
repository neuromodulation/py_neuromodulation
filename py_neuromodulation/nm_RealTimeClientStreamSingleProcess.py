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
    esc_key_received : bool = False
    use_FieldTripClient : bool = True
    feature_count_idx : int = 0

    def __init__(
        self,
        PATH_SETTINGS: str,
        PATH_NM_CHANNELS: str,
        PATH_OUT: str = os.getcwd(),
        filename: str = "testrun",
        PATH_GRIDS: str = pathlib.Path(__file__).parent.resolve(),
        VERBOSE: bool = True,
        fs: int = 128,
        line_noise: int = 50,
        use_FieldTripClient : bool = True
    ) -> None:

        super().__init__(PATH_SETTINGS=PATH_SETTINGS,
            PATH_NM_CHANNELS=PATH_NM_CHANNELS,
            PATH_OUT=PATH_OUT,
            PATH_GRIDS=PATH_GRIDS,
            VERBOSE=VERBOSE)

        self.set_fs(fs)
        self.set_linenoise(line_noise)
        self.filename = filename
        self.use_FieldTripClient = use_FieldTripClient

        self.nm_channels = self._get_nm_channels(
            self.PATH_NM_CHANNELS
        )

        # leave out coordinate setting for now

        self._set_run()

        if self.use_FieldTripClient is True:
            self.ftc = self.init_fieldtrip()
            self.get_data_client = self.get_data_FieldTripClient
            self.send_data_client = self.send_data_FieldTripClient
            self.disconnect = self.disconnect_FieldTripClient
        else:
            # use LSL
            self.get_data_client = None
            self.send_data_client = None
            self.disconnect = None
        
        self.init_keyboard_listener()

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
        last_batch = 0

        while self.listener.is_alive() is True:
            ieeg_batch = self.get_data_client()
            ieeg_batch = ieeg_batch[-128:, :2]  # take last, #1: data
            # check if time stamp changed 
            if np.array_equal(ieeg_batch, last_batch) is False:

                last_batch = ieeg_batch

                feature_series = \
                    self.run_analysis.process_data(ieeg_batch)
                feature_series = self._add_timestamp(feature_series)

                if self.esc_key_received is True:
                    self._sendFeatures(features=None)
                else:
                    self._sendFeatures(feature_series)

        self.disconnect()

    def get_data_FieldTripClient(self):
        return self.ftc.getData()

    def send_data_FieldTripClient(self, features: pd.Series):

        H = self.ftc.getHeader()
        # channel names are 1. data 2. ident 3. timestamp
        # put at the end of the buffer the calculated features of the data channel
        to_send = np.zeros([H.nSamples, H.nChannels])
        to_send[-features.shape[0]:, 1] = np.array(features)

        self.ftc.putData(to_send)

    def disconnect_FieldTripClient(self):
        self.ftc.disconnect()

    def init_keyboard_listener(
        self
    ):

        def on_press(key):
            print('{0} pressed'.format(
                key))

        def on_release(key):
            if key == Key.esc:
                # Stop listener
                print("reiceived stop key pressed")
                self.esc_key_received = True
                # terminate session and save feature arr
                self._sendFeatures(None)
                return False

        self.listener = Listener(
            on_press=on_press,
            on_release=on_release)
        self.listener.start()
    
    def _sendFeatures(self, features: pd.Series):
        if features is None:
            self.save_after_stream(self.filename)
            print("SAVED")
            return True
        else:
            if self.feature_count_idx == 0:
                self.feature_arr = pd.DataFrame([features])
                self.feature_count_idx += 1
            else:
                self.feature_arr = self.feature_arr.append(
                    features,
                    ignore_index=True
                )

            print("length of features:" + str(len(self.feature_arr)))

            self.send_data_client(features)
        return False

    def _add_timestamp(self, feature_series: pd.Series, idx: int = None) -> pd.Series:

        feature_series["time"] = time.time()  # UNIX Timestamp

        return feature_series

    def _add_coordinates(self):
        """Lateron add here method for providing coordinates"""
        pass

    def get_data(self) -> np.array:
        pass
