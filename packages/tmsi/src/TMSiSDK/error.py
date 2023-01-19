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
 * @file ${error.py} 
 * @brief General SDK errors
 *
 */


"""
from enum import Enum, unique


@unique
class TMSiErrorCode(Enum):
    general_error = 0
    missing_dll = 1
    already_in_use_dll = 2
    device_error = 100
    device_not_connected = 101
    no_devices_found = 102
    api_no_driver = 200
    api_incorrect_argument = 201
    api_invalid_command = 202
    file_writer_error = 300


class TMSiError(Exception):
    def __init__(self, code):
        self.code = code

    def __str__(self):
        return repr(self.code)


def DeviceErrorLookupTable(code):
    _lookup_table = {
        "0x1010001": "DR reported 'Checksum error in received block'",
        "0x2010001": "DS reported 'Checksum error in received block'",
        "0x1010002": "DR reported 'Unknown command'",
        "0x2010002": "DS reported 'Unknown command'",
        "0x1010003": "DR reported 'Response timeout'",
        "0x2010003": "DS reported 'Response timeout'",
        "0x1010004": "DR reported 'Device busy try again in x msec'",
        "0x2010004": "DS reported 'Device busy try again in x msec'",
        "0x1010005": "DR reported 'Command not supported over current interface'",
        "0x2010005": "DS reported 'Command not supported over current interface'",
        "0x1010006": "DR reported 'Command not possible, device is recording'",
        "0x1010007": "DR reported 'Device not available'",
        "0x2010007": "DS reported 'Device not available'",
        "0x2010008": "DS reported 'Interface not available'",
        "0x2010009": "DS reported 'Command not allowed in current mode'",
        "0x201000A": "DS reported 'Processing error'",
        "0x201000B": "DS reported 'Unknown internal error'",
        "0x1030001": "DR reported 'Command not supported by Channel'",
        "0x1030002": "DR reported 'Illegal start control for ambulant recording",
        "0x201000C": "DS reports that data request does not fit with one Device Api Packet",
        "0x201000D": "DS reports that DR is already opened",
        "0x3001000": "DLL Function is declared, but not yet implemented",
        "0x3001001": "DLL Function called with invalid parameters",
        "0x3001002": "Incorrect checksum of response message",
        "0x3001003": "DLL Function failed because of header failure",
        "0x3001004": "DLL Function failed because an underlying process failed",
        "0x3001005": "DLL Function called with a too small buffer",
        "0x3001006": "DLL Function called with a Handle that's not assigned to a device",
        "0x3002000": "DLL Function failed becasue could not open selected interface",
        "0x3002001": "DLL Function failed because could not close selected interface",
        "0x3002002": "DLL Function failed because could not send command-data",
        "0x3002003": "DLL Function failed because could not receive data",
        "0x3002004": "DLL Function failed because commination timed out",
        "0x3002005": "Lost connection to DS, USB / Ethernet disconnect",
    }

    print("\n " + _lookup_table[code] + "\n")
