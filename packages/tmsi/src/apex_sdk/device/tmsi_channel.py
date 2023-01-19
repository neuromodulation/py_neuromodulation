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
 * @file tmsi_channel.py 
 * @brief 
 * TMSi Channel interface.
 */


"""

from .tmsi_device_enums import ChannelType


class TMSiChannel:
    def __init__(self):
        self._alt_name = "-"
        self._chan_divider = -1
        self._def_name = "-"
        self._enabled = False
        self._exp = 0
        self._format = 0
        self._is_reference = 0
        self._imp_divider = -1
        self._sample_rate = 0
        self._type = ChannelType.unknown
        self._unit_name = "-"

    def get_channel_exp(self):
        return self._exp

    def get_channel_format(self):
        return self._format

    def get_channel_name(self):
        return self._alt_name

    def get_channel_type(self):
        return self._type

    def get_channel_unit_name(self):
        return self._unit_name

    def is_reference(self):
        if self._is_reference == 1:
            return 1
        else:
            return 0

    def set_device_channel_information(self):
        raise NotImplementedError("method not available for this channel")

    def set_device_channel_names(
        self, default_channel_name=None, alternative_channel_name=None
    ):
        if default_channel_name is not None:
            self._def_name = default_channel_name
        if alternative_channel_name is not None:
            self._alt_name = alternative_channel_name

    def set_device_reference(self, is_reference):
        self._is_reference = is_reference
