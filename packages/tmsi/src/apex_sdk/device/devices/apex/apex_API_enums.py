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
 * @file apex_API_enums.py 
 * @brief 
 * Enumerable classes useful for the use of the API.
 */


'''

from enum import Enum, unique
#---------------------------------------------------------------------

# Error codes

#---
# 0x01xxxxxx = FE API related, 0x02xxxxxx is reserved for USER API
# Error codes are categorized as:
# 0x0101xxxx # Gen. System status
# 0x0102xxxx # Hardware related status
# 0x0103xxxx # Firmware related status
#---

#---
# Defined status codes are:
# Generic Device status codes for the DR 0x0101xxxx
# Generic Device status codes for the DS 0x0201xxxx
#
# Hardware specific status codes for the DR 0x0102xxxx
# Hardware specific status codes for the DS 0x0202xxxx
#
# Firmware specific status codes for the DR 0x0103xxxx
# Firmware specific status codes for the DS 0x0203xxxx
#
#---
# Each DLL API function on the TMSi Device API has a return value TMSiDeviceRetVal.
@unique
class TMSiDeviceRetVal(Enum):
    TMSiStatusOK                            = 0x00000000  # All Ok positive ACK
    TMSiStatusDrChecksumError               = 0x01010001  # DR reports "Checksum error in received block"
    TMSiStatusDrUnknownCommand              = 0x01010002  # DR reports "Unknown command"
    TMSiStatusDrResponseTimeout             = 0x01010003  # DR reports "Response timeout"
    TMSiStatusDrInterfaceAlreadyOpen        = 0x01010004  # DR reports "Interface is already open"
    TMSiStatusDrUnknownCommandForInterface  = 0x01010005  # DR reports "Command not supported over current interface"
    TMSiStatusDrDeviceRecording             = 0x01010006  # DR reports "Command not possible, device is recording"
    TMSiStatusDrConfigError                 = 0x01010007  # DR reports "Configuration Error"
    TMSiStatusDrDeviceLocked                = 0x01010008  # DR reports "Device is locked for use"
    TMSiStatusDrServiceModeLocked           = 0x01010009  # DR reports "Device service mode is locked"

    # Additional defines below for DLL error types.
    TMSiStatusDllDeviceNotAvailable         = 0x03001001  # DLL reports "Device not available".
    TMSiStatusDllInterfaceAlreadyOpen       = 0x03001002  # DLL reports that interface is already opened.
    TMSiStatusDllDeviceNotPaired            = 0x03001003  # DLL reports that it is not paired
    TMSiStatusDllDeviceAlreadyPaired        = 0x03001004  # DLL reports that it is already paired.
    TMSiStatusDllNotImplemented             = 0x03001005  # DLL Function is declared, but not yet implemented
    TMSiStatusDllInvalidParameter           = 0x03001006  # DLL Function called with invalid parameters
    TMSiStatusDllChecksumError              = 0x03001007
    TMSiStatusDllInternalError              = 0x03001008  # Function failed because an underlying process failed
    TMSiStatusDllBufferError                = 0x03001009  # Function called with a too small buffer
    TMSiStatusDllInvalidHandle              = 0x0300100A  # Function called with a Handle that's not assigned to a device
    TMSiStatusDllInterfaceOpenError         = 0x0300100B
    TMSiStatusDllInterfaceCloseError        = 0x0300100C
    TMSiStatusDllInterfaceSendError         = 0x0300100E
    TMSiStatusDllInterfaceReceiveError      = 0x0300100F
    TMSiStatusDllInterfaceTimeout           = 0x03001010
    TMSiStatusDllCommandInProgress          = 0x03001011
    TMSiStatusDllNoEventAvailable           = 0x03001012
    TMSiStatusDllInvalidCardFileID          = 0x03001013
    TMSiStatusDllCanNotDecodeSampleData     = 0x03001014


# Communication interface used
# 0 = Unknown, 1=USB 2=Nework, 3=WiFi, 4=Electrical, 5=Optical, 6=Bluetooth, 7=Serial.
@unique
class TMSiInterface(Enum):
    IfTypeUnknown = 0    # Unknown interface type.
    IfTypeUSB = 1        # USB interface.
    IfTypeNetwork = 2    # Wired Ethernet interface.
    IfTypeWiFi = 3       # Wireless 802.11x interface.
    IfTypeElectrical = 4 # Proprietary electrical interface.
    IfTypeOptical = 5    # Proprietary optical interface.
    IfTypeBluetooth = 6  # Bluetooth interface.
    IfTypeSerial = 7     # UART serial interface.

# The pairing status between the dongle and the recorder
@unique
class TMSiPairingStatus(Enum):
    PairingStatusNoPairingNeeded = 0
    PairingStatusPaired = 1
    PairingStatusNotPaired = 2

#
@unique
class TMSiControl(Enum):
    ControlDisabled = 0
    ControlEnabled = 1

#
@unique
class TMSiInterfaceControlStatus(Enum):
    InterfaceOffline = 0
    InterfaceOnline = 1

#
@unique
class TMSiInterfaceControlStatus(Enum):
    InterfaceOffline = 0
    InterfaceOnline = 1

#
@unique
class TMSiChannelRefStatus(Enum):
    ChannelNotReference = 0
    ChannelInReference = 1

#
@unique
class SampleControl(Enum):
    StopSampling = 0
    StartSampling = 1

#
@unique
class ImpedanceControl(Enum):
    StopImpedance = 0
    StartImpedance = 1


#
@unique
class SampleData(Enum):
    SamplingData = 1
    ImpedanceData = 2
    CardRecordingData = 3

#
@unique
class CardRecDownloadControl(Enum):
    CardRecDownloadStop = 0
    CardRecDownloadStart = 1

#
@unique
class TMSiEventID(Enum):
    EventIdConnection = 0
    EventIdPairing = 1


#
@unique
class TMSiEventCode(Enum):
    EventCodeOK = 0
    EventCodeFailure = 1

#
@unique
class TMSiBaseSampleRate(Enum):
    Decimal = 1000
    Binary = 1024

@unique
class TMSiLiveImpedance(Enum):
    On = 1
    Off = 0

@unique
class ApexStringLengths(Enum):
    AltChanName = 10
    PrefixFileName = 16
    UserString = 64

@unique
class ApexStartCardRecording(Enum):
    Time = 1
    Button = 8