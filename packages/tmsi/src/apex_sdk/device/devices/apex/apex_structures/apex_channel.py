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
 * @file apex_channel.py 
 * @brief 
 * APEX Channel object.
 */


"""

from ....tmsi_channel import TMSiChannel, ChannelType


class ApexChannel(TMSiChannel):
    def set_device_channel_information(self, channel_metadata):
        self._type = ChannelType(channel_metadata.ChannelType)
        self._format = channel_metadata.ChannelFormat
        self._chan_divider = channel_metadata.ChanDivider
        self._enabled = self._chan_divider != -1
        self._imp_divider = channel_metadata.ImpDivider
        self._exp = channel_metadata.Exp
        self._unit_name = channel_metadata.UnitName.decode("windows-1252")
