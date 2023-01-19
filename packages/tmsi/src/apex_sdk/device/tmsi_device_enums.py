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
 * @file tmsi_device_enums.py 
 * @brief 
 * Enumerable classes useful for the use of the SDK.
 */


'''

from enum import Enum, unique
from .devices.apex.measurements.eeg_measurement import EEGMeasurement
from .devices.apex.measurements.impedance_measurement import ImpedanceMeasurement
from .devices.apex.measurements.download_measurement import DownloadMeasurement


@unique
class DeviceInterfaceType(Enum):
    none = 0
    usb = 1
    network = 2
    wifi = 3
    docked = 4
    optical = 5
    bluetooth = 6
    serial = 7

@unique
class DeviceState(Enum):
    disconnected = 0
    connected = 1
    sampling = 2

@unique
class PairingStatus(Enum):
    no_pairing_needed = 0
    paired = 1
    not_paired = 2	

@unique
class ChannelType(Enum):
    unknown = 0
    UNI = 1
    BIP = 2
    AUX = 3
    sensor = 4
    status = 5
    counter = 6
    impedance = 7
    cycling_status = 8
    all_types = 9

@unique
class ReferenceMethod(Enum):
    common = 0
    average = 1

@unique
class DeviceType(Enum):
    none = 0
    saga = 1
    apex = 2

@unique
class ReferenceSwitch(Enum):
    fixed=0
    auto=1

class MeasurementType():
    APEX_EEG = EEGMeasurement
    APEX_IMPEDANCE = ImpedanceMeasurement
    APEX_DOWNLOAD = DownloadMeasurement

