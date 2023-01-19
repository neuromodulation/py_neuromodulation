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
 * @file sampling_thread.py 
 * @brief 
 * Sampling Thread interface
 */


'''

import threading
import time


class SamplingThread(threading.Thread):
    def __init__(self, sampling_function, pause = 0.01, name = "Sampling Thread"):
        super().__init__()
        self.__name = name
        self.__sampling_function = sampling_function
        self.__pause = pause

    def get_pause(self):
        return self.__pause
    
    def run(self):
        self.__sampling = True
        while self.__sampling:
            self.__sampling_function()
            time.sleep(self.__pause)

    def set_pause(self, pause):
        self.__pause = pause

    def stop(self):
        self.__sampling = False