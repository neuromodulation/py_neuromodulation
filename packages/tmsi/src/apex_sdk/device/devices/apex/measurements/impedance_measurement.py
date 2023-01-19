'''
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
 * @file impedance_measurement.py 
 * @brief 
 * Class to handle the communication with the device for  impedance acquisition.
 */


'''

from copy import deepcopy
from ctypes import *
import time

from .....tmsi_utilities.tmsi_logger import TMSiLogger as logger
from .....sample_data_server.sample_data_server import SampleDataServer
from .....sample_data_server.sample_data import SampleData
from .....tmsi_utilities.decorators import LogPerformances
from ..apex_API_structures import TMSiImpedanceSample, TMSiDevImpedanceRequest
from ..apex_API_enums import ImpedanceControl, TMSiDeviceRetVal
from ....tmsi_measurement import TMSiMeasurement


class ImpedanceMeasurement(TMSiMeasurement):
    def __init__(self, dev, name = "Impedance Measurement"):
        super().__init__(dev = dev, name = name)
        self._sample_data_buffer = (TMSiImpedanceSample * self._sample_data_buffer_size)(TMSiImpedanceSample())
        self._num_samples_per_set = dev.get_num_impedance_channels()
        
    @LogPerformances
    def _sampling_function(self):
        ret = self._dev.get_device_impedance_data(
            pointer(self._sample_data_buffer), 
            self._sample_data_buffer_size, 
            pointer(self._retrieved_sample_sets))
        if (ret == TMSiDeviceRetVal.TMSiStatusOK):
                if self._retrieved_sample_sets.value > 0:
                    self._conversion_queue.put((deepcopy(self._sample_data_buffer), self._retrieved_sample_sets.value))
                    self._empty_read_counter = 0
                else:
                    if self._empty_read_counter == 0:
                        self._tic_timeout = time.perf_counter()
                    self._empty_read_counter += 1

    @LogPerformances
    def _conversion_function(self):
        while not self._conversion_queue.empty():
            sample_data_buffer, retrieved_sample_sets = self._conversion_queue.get()
            if retrieved_sample_sets > 0:
                sample_mat = [sample_data_buffer[ii:retrieved_sample_sets * self._num_samples_per_set:self._num_samples_per_set] 
                    for ii in range(self._num_samples_per_set)]
                samples_exploded =  [
                    [i.ImpedanceRe if ii%2==0 else i.ImpedanceIm for i in sample_mat[ii//2]] 
                    for ii in range(self._num_samples_per_set * 2)]
                samples_exploded_inline = [samples_exploded[i][j] for j in range(len(samples_exploded[0])) for i in range(len(samples_exploded))]
                sd = SampleData(retrieved_sample_sets, self._num_samples_per_set * 2, samples_exploded_inline)
                logger().debug("Data delivered to sample data server: {} channels, {} samples".format(self._num_samples_per_set, retrieved_sample_sets))
                SampleDataServer().put_sample_data(self._dev.get_id(), sd)

    @LogPerformances
    def start(self):
        self._dev.reset_device_data_buffer()
        measurement_request = TMSiDevImpedanceRequest()
        measurement_request.StartStop = ImpedanceControl.StartImpedance.value
        self._dev.set_device_impedance_request(measurement_request)
        self._sampling_thread.start()
        self._conversion_thread.start()

    @LogPerformances
    def stop(self):
        measurement_request = TMSiDevImpedanceRequest()
        measurement_request.StartStop = ImpedanceControl.StopImpedance.value
        self._dev.set_device_impedance_request(measurement_request)
        self._sampling_thread.stop()
        self._conversion_thread.stop()