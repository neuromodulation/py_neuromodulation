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
 * @file apex_info.py 
 * @brief 
 * APEX Information object.
 */


"""

from ....tmsi_device_enums import *

from .apex_const import ApexConst


class ApexInfo:
    def __init__(
        self,
        dongle_serial_number=0,
        serial_number=0,
        id=ApexConst.TMSI_DEVICE_ID_NONE,
        dr_interface=DeviceInterfaceType.none,
        pairing_status=PairingStatus.no_pairing_needed,
    ):
        self.__dr_interface = dr_interface
        self.__dr_serial_number = serial_number
        self.__id = id
        self.__dongle_id = ApexConst.TMSI_DONGLE_ID_NONE
        self.__state = DeviceState.disconnected
        self.__pairing_status = pairing_status
        self.__dongle_serial_number = dongle_serial_number
        self.__num_hw_channels = 0
        self.__num_cycling_states = 0
        self.__num_imp_channels = 0
        self.__num_channels = 0

    def get_dr_interface(self):
        return self.__dr_interface

    def get_id(self):
        return self.__id

    def get_num_channels(self):
        return self.__num_channels

    def get_num_impedance_channels(self):
        return self.__num_imp_channels

    def get_dongle_serial_number(self):
        return self.__dongle_serial_number

    def get_dr_serial_number(self):
        return self.__dr_serial_number

    def get_pairing_status(self):
        return self.__pairing_status

    def get_state(self):
        return self.__state

    def set_device_info_report(self, device_info_report):
        self.__num_channels = device_info_report.NrOfChannels
        self.__num_hw_channels = device_info_report.NrOfHWChannels
        self.__num_imp_channels = device_info_report.NrOfImpChannels
        self.__num_cycling_states = device_info_report.NrOfCyclingStates

    def set_dongle_id(self, dongle_id):
        self.__dongle_id = dongle_id

    def set_dongle_serial_number(self, dongle_serial_number):
        self.__dongle_serial_number = dongle_serial_number

    def set_dr_interface(self, dr_interface):
        self.__dr_interface = dr_interface

    def set_dr_serial_number(self, dr_serial_number):
        self.__dr_serial_number = dr_serial_number

    def set_id(self, id):
        self.__id = id

    def set_pairing_status(self, pairing_status):
        self.__pairing_status = pairing_status

    def set_state(self, state):
        self.__state = state
