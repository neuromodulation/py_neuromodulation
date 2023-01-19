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
 * @file tmsi_event_reader.py 
 * @brief 
 * Event reader interface.
 */


"""

from .threads.sampling_thread import SamplingThread


class TMSiEventReader:
    def __init__(self, name="Event Reader"):
        self._name = name
        self._reading_thread = SamplingThread(
            sampling_function=self._reading_function, pause=0.5
        )

    def start(self):
        raise NotImplementedError("method not available for this reader")

    def stop(self):
        self._reading_thread.stop()

    def _reading_function(self):
        raise NotImplementedError("method not available for this reader")
