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
 * @file apex_event_reader.py 
 * @brief 
 * APEX Event reader.
 */


"""

from ctypes import c_ushort, pointer

from ....sample_data_server.sample_data_server import SampleDataServer
from ...tmsi_event_reader import TMSiEventReader
from .apex_device import ApexDevice, TMSiEvent
from ....tmsi_utilities.tmsi_logger import TMSiLogger


class ApexEventReader(TMSiEventReader):
    def __init__(self, name="Apex Event Reader"):
        super().__init__(name)

    def start(self):
        ApexDevice.reset_device_event_buffer()
        self._reading_thread.start()

    def _reading_function(self):
        num_events = (c_ushort)(0)
        ApexDevice.get_event_buffer(pointer(num_events))
        while num_events.value > 0:
            TMSiLogger().warning("An event occurred")
            event = TMSiEvent()
            ApexDevice.get_event(pointer(event))
            SampleDataServer().put_event_data(id=event.TMSiDeviceID, data=event)
            num_events.value -= 1
