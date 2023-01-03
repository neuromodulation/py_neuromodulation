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

# import pylsl
#  clone TMSI-Python-Interface from here: https://gitlab.com/tmsi/tmsi-python-interface

from py_neuromodulation import (
    nm_projection,
    nm_rereference,
    nm_run_analysis,
    nm_features,
    nm_resample,
    nm_settings,
    nm_stream,
    FieldTrip,
)


class RealTimePyNeuro(nm_stream.PNStream):

    queue_raw: multiprocessing.Queue = multiprocessing.Queue(1)
    queue_features: multiprocessing.Queue = multiprocessing.Queue(1)
    ftc: FieldTrip.Client = None
    filename: str
    esc_key_received: bool = False
    use_FieldTripClient: bool = True
    feature_count_idx: int = 0

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
        use_FieldTripClient: bool = True,
    ) -> None:

        super().__init__(
            path_settings=PATH_SETTINGS,
            path_nm_channels=PATH_NM_CHANNELS,
            path_out=PATH_OUT,
            path_grids=PATH_GRIDS,
            verbose=VERBOSE,
        )

        self.set_fs(fs)
        self.set_linenoise(line_noise)
        self.filename = filename
        self.use_FieldTripClient = use_FieldTripClient

        self.nm_channels = self._get_nm_channels(self.path_nm_channels)

        # leave out coordinate setting for now

        if self.use_FieldTripClient is True:
            self.ftc = self.init_fieldtrip(1972)
            self.ftc_send = self.init_fieldtrip(1987)
            self.get_data_client = self.get_data_FieldTripClient
            self.send_data_client = self.send_data_FieldTripClient
            self.disconnect = self.disconnect_FieldTripClient
        else:
            # use LSL
            self.lsl_client = self.init_lsl(wait_max=10, buffer_size=1000)
            self.get_data_client = self.get_data_lsl
            self.send_data_client = self.send_data_lsl
            self.disconnect = self.disconnect_lsl

        self.init_keyboard_listener()

    @staticmethod
    def init_fieldtrip(PORT: int = 1972) -> FieldTrip.Client:
        """Initialize Fieldtrip client
        Returns
        -------
        FieldTrip.Client
        """
        ftc = FieldTrip.Client()
        # Python FieldTripBuffer https://www.fieldtriptoolbox.org/development/realtime/buffer_python/
        ftc.connect("localhost", port=PORT)  # might throw IOError
        H = ftc.getHeader()

        if H is None:
            print("Failed to retrieve header!")
            sys.exit(1)

        print(H)
        print(H.labels)

        return ftc

    @staticmethod
    def init_lsl(wait_max: int = 10, buffer_size: int = 1000):

        # streams = pylsl.resolve_streams(wait_time=min(0.1, wait_max))
        # print("Stream found")
        # return pylsl.StreamInlet(info=streams[0], max_buflen=buffer_size)
        ...

    def run(self) -> None:
        """Start get_data, calcFeatures and sendFeature processes"""

        self._set_run()

        last_batch = 0

        while self.listener.is_alive() is True:
            ieeg_batch = self.get_data_client()
            ieeg_batch = ieeg_batch[-128:, :2]  # take last, #1: data
            # ieeg_batch = np.random.random([128, 2]).T  # channels, samples
            # check if time stamp changed
            if np.array_equal(ieeg_batch, last_batch) is False:

                last_batch = ieeg_batch

                feature_series = self.run_analysis.process_data(ieeg_batch)
                feature_series = self._add_timestamp(feature_series)

                if self.esc_key_received is True:
                    self._sendFeatures(features=None)
                else:
                    self._sendFeatures(feature_series)

        self.disconnect()

    def get_data_lsl(self, max_samples: int = 1000, timeout: int = 5):
        # samples, _ = self.lsl_client.pull_chunk(
        #    max_samples=max_samples,
        #    timeout=timeout
        # )
        # return np.vstack(samples).T
        ...

    def send_data_lsl(self, features: pd.Series):
        """Not tested yet
        Check https://github.com/mne-tools/mne-realtime/blob/main/mne_realtime/mock_lsl_stream.py
        for reference implementation
        """

        # info = pylsl.StreamInfo(name='py_nm', type=None,
        #                        channel_count=None,
        #                        nominal_srate=None,
        #                        channel_format='float32', source_id=None)
        # outlet = pylsl.StreamOutlet(info)
        # outlet.push_sample(features.to_numpy())
        ...

    def disconnect_lsl(self):
        self.lsl_client.close_stream()

    def get_data_FieldTripClient(self):
        return self.ftc.getData()

    def send_data_FieldTripClient(self, features: pd.Series):

        # H = self.ftc_send.getHeader()  # retrieving header is not necessary for sending
        self.ftc_send.putData(np.random.random([1, 2]))
        # self.ftc_send.putData(np.expand_dims(np.array(features), axis=0))

    def disconnect_FieldTripClient(self):
        self.ftc.disconnect()

    def init_keyboard_listener(self):
        def on_press(key):
            print("{0} pressed".format(key))

        def on_release(key):
            if key == Key.esc:
                # Stop listener
                print("reiceived stop key pressed")
                self.esc_key_received = True
                # terminate session and save feature arr
                self._sendFeatures(None)
                return False

        self.listener = Listener(on_press=on_press, on_release=on_release)
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
                    features, ignore_index=True
                )

            print("length of features:" + str(len(self.feature_arr)))
            try:
                self.send_data_client(features)
            except IOError:
                print("IOError writing data back to Client")

        return False

    def _add_timestamp(
        self, feature_series: pd.Series, idx: int = None
    ) -> pd.Series:

        feature_series["time"] = time.time()  # UNIX Timestamp

        return feature_series

    def _add_coordinates(self):
        """Lateron add here method for providing coordinates"""
        pass

    def get_data(self) -> np.array:
        pass
