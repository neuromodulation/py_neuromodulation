"""
(c) 2022 Twente Medical Systems International B.V., Oldenzaal The Netherlands

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

#######  #     #   #####   #
   #     ##   ##  #        
   #     # # # #  #        #
   #     #  #  #   #####   #
   #     #     #        #  #
   #     #     #        #  #
   #     #     #  #####    #

/**
 * @file tmsi_measurement.py 
 * @brief 
 * Measurement interface.
 */


"""

from ctypes import *
import queue
import time

from ..tmsi_utilities.decorators import LogPerformances
from ..tmsi_utilities.tmsi_logger import TMSiLogger as logger
from .threads.sampling_thread import SamplingThread
from .threads.conversion_thread import ConversionThread


class TMSiMeasurement:
    def __init__(self, dev, name="TMSi Measurement"):
        self._dev = dev
        self._name = name
        self._downloaded_samples = 0
        self._download_percentage = 0
        self._download_samples_limit = None
        self._sample_data_buffer_size = 409600
        self._retrieved_sample_sets = (c_uint)(0)
        self._retrieved_data_type = (c_int)(0)
        __MAX_SIZE_CONVERSION_QUEUE = 50
        self._conversion_queue = queue.Queue(__MAX_SIZE_CONVERSION_QUEUE)
        self._empty_read_counter = 0
        self._tic_timeout = None
        self._timeout = 3
        self._disable_live_impedance = 0
        self._disable_average_reference_calculation = 0
        self._float_channels = []
        channels = self._dev.get_device_channels()
        for i in range(len(channels)):
            if channels[i].get_channel_format() == 0x0020:
                self._float_channels.append(i)
        self._sampling_thread = SamplingThread(
            sampling_function=self._sampling_function, pause=0.01
        )
        self._conversion_thread = ConversionThread(
            conversion_function=self._conversion_function, pause=0.01
        )

    @LogPerformances
    def get_conversion_pause(self):
        return self._conversion_thread.get_pause()

    @LogPerformances
    def get_sampling_pause(self):
        return self._sampling_thread.get_pause()

    @LogPerformances
    def is_timeout(self):
        if self._tic_timeout is None:
            return False
        return self._timeout < time.perf_counter() - self._tic_timeout

    @LogPerformances
    def get_download_percentage(self):
        return self._download_percentage

    @LogPerformances
    def set_conversion_pause(self, pause):
        logger().debug("conversion pause set to {}".format(pause))
        self._conversion_thread.set_pause(pause)

    @LogPerformances
    def set_download_samples_limit(self, max_number_of_samples=None):
        self._download_samples_limit = max_number_of_samples - 1

    @LogPerformances
    def set_sampling_pause(self, pause):
        logger().debug("sampling pause set to {}".format(pause))
        self._sampling_thread.set_pause(pause)

    @LogPerformances
    def start(self):
        raise NotImplementedError("method not available for this measurement")

    @LogPerformances
    def stop(self):
        raise NotImplementedError("method not available for this measurement")

    @LogPerformances
    def _conversion_function(self):
        raise NotImplementedError("method not available for this measurement")

    @LogPerformances
    def _sampling_function(self):
        raise NotImplementedError("method not available for this measurement")
