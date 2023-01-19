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
 * @file download_measurement.py 
 * @brief 
 * Class to handle the download of a file from the device.
 */


'''

from .....tmsi_utilities.decorators import LogPerformances
from ..apex_API_structures import TMSiDevSetCardFileReq
from ..apex_API_enums import SampleControl
from .eeg_measurement import EEGMeasurement


class DownloadMeasurement(EEGMeasurement):
    def __init__(self, dev, file_id: int, n_of_samples: int = None, name:str = "Download Measurement"):
        super().__init__(dev, name)
        self._file_id = file_id
        if n_of_samples is None:
            header, metadata = self._dev.get_device_card_file_info(self._file_id)
            self._n_of_samples = metadata.NumberOfSamples
        else:
            self._n_of_samples = n_of_samples
        self.set_download_samples_limit(self._n_of_samples)

    @LogPerformances
    def start(self):
        self._dev.reset_device_data_buffer()
        file_request = TMSiDevSetCardFileReq()
        file_request.RecFileID = self._file_id
        file_request.StartCounter = 0
        file_request.NumberOfSamples = self._n_of_samples
        file_request.StartStop = SampleControl.StartSampling.value
        self._dev.set_device_download_file_request(file_request)
        self._sampling_thread.start()
        self._conversion_thread.start()

    @LogPerformances
    def stop(self):
        file_request = TMSiDevSetCardFileReq()
        file_request.RecFileID = self._file_id
        file_request.StartStop = SampleControl.StopSampling.value
        self._dev.set_device_download_file_request(file_request)
        self._sampling_thread.stop()
        self._conversion_thread.stop()