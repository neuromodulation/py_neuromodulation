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
 * @file apex_impedance_channel.py 
 * @brief 
 * APEX Impedance Channel object.
 */


'''

class ApexImpedanceChannel:
    def __init__(self, channel_metadata):
        self.set_device_impedance_channel_metadata(channel_metadata)

    def get_channel_name(self):
        return self.__channel_name

    def get_channel_unit_name(self):
        return (self.__impedance_re_unit, self.__impedance_im_unit)
    
    def set_device_impedance_channel_metadata(self, channel_metadata):
        self.__channel_index = channel_metadata.ChanIdx
        self.__channel_name = channel_metadata.ChanName.decode()
        self.__impedance_im_unit = channel_metadata.ImpedanceImUnit.decode()
        self.__impedance_re_unit = channel_metadata.ImpedanceReUnit.decode()