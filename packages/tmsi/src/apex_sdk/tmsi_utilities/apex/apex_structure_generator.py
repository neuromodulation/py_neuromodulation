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
 * @file apex_structure_generator.py 
 * @brief 
 * Object that creates the structures needed to communicate with APEX.
 */


'''

import datetime

from ...device.devices.apex import apex_API_structures as ApexStructures
from ...device.devices.apex import apex_API_enums as ApexEnums
from ...device.devices.apex.apex_device import ApexDevice


class ApexStructureGenerator:
    def create_card_record_configuration(
        device: ApexDevice,
        start_control: ApexEnums.ApexStartCardRecording,
        prefix_file_name: str = None,
        start_time: datetime.datetime = None,
        stop_time: datetime.datetime = None,
        pre_measurement_imp = None,
        pre_measeurement_imp_seconds = None,
        user_string_1: str = None,
        user_string_2: str = None
    ) -> ApexStructures.TMSiDevCardRecCfg:
        """Creates the TMSiDevCardRecCfg structure with provided parameters

        :param device: device to pull the configuration from
        :type device: ApexDevice
        :param prefix_file_name: prefix file name, defaults to None.
        :type prefix_file_name: str, optional
        :param start_time: datetime of the start, defaults to None.
        :type start_time: datetime.datetime, optional
        :param stop_time: datetime of the stop, defaults to None.
        :type stop_time: datetime.datetime, optional
        :param user_string_1: user string, defaults to None.
        :type user_string_1: str, optional
        :param user_string_2: user string, defaults to None.
        :type user_string_2: str, optional
        :return: the structure containing provided information
        :rtype: ApexStructures.TMSiDevCardRecCfg
        """
        
        config = device.get_card_recording_config()
        config.StartControl = start_control.value
        if prefix_file_name is not None:
            converted_str = bytearray(ApexEnums.ApexStringLengths.PrefixFileName.value)
            for character in range(len(prefix_file_name)):
                converted_str[character] = ord(prefix_file_name[character])
            config.PrefixFileName = bytes(converted_str)
        if user_string_1 is not None:
            converted_str = bytearray(ApexEnums.ApexStringLengths.UserString.value)
            for character in range(len(user_string_1)):
                converted_str[character] = ord(user_string_1[character])
            config.UserString1 = bytes(converted_str)
        if user_string_2 is not None:
            converted_str = bytearray(ApexEnums.ApexStringLengths.UserString.value)
            for character in range(len(user_string_2)):
                converted_str[character] = ord(user_string_2[character])
            config.UserString2 = bytes(converted_str)
        if start_time is not None:
            ApexStructureGenerator.from_datetime_to_tmsitime(
                start_time, config.StartTime)
        if stop_time is not None:
            ApexStructureGenerator.from_datetime_to_tmsitime(
                stop_time, config.StopTime)
        if pre_measurement_imp is not None:
            config.PreImp = pre_measurement_imp
        if pre_measeurement_imp_seconds is not None:
            config.PreImpSec = pre_measeurement_imp_seconds
        return config

    def from_qdatetime_to_tmsitime(qdatetime, tmsi_time):
        tmsi_time.Seconds = qdatetime.time().second()
        tmsi_time.Minutes = qdatetime.time().minute()
        tmsi_time.Hours = qdatetime.time().hour()
        tmsi_time.DayOfMonth = qdatetime.date().day()
        tmsi_time.Month = qdatetime.date().month() - 1
        tmsi_time.Year = qdatetime.date().year()-1900
        return tmsi_time

    def from_datetime_to_tmsitime(date_time, tmsi_time):
        tmsi_time.Seconds = date_time.time().second
        tmsi_time.Minutes = date_time.time().minute
        tmsi_time.Hours = date_time.time().hour
        tmsi_time.DayOfMonth = date_time.date().day
        tmsi_time.Month = date_time.date().month - 1
        tmsi_time.Year = date_time.date().year-1900
        return tmsi_time

    
    def from_tmsitime_to_datetime(tmsi_time, date_time):
        date_time = datetime.datetime(
            tmsi_time.Year + 1900,
            tmsi_time.Month + 1,
            tmsi_time.DayOfMonth,
            tmsi_time.Hours,
            tmsi_time.Minutes,
            tmsi_time.Seconds)
        return date_time

    