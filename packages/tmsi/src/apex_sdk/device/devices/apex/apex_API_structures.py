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
 * @file apex_API_structures.py 
 * @brief 
 * Structures useful for the use of the API.
 */


'''

from ctypes import *
# Protocol block definitions

#---
# TMSiDevList
#---
class TMSiDevList(Structure):
    _pack_=1
    _fields_ = [
        ("TMSiDeviceID", c_ushort),             # Unique ID to identify device, used to open device.
        ("SerialNumberDataRecorder", c_uint),   # The Data Recorder serial number.
        ("PairingStatus", c_ubyte),             # The pairing status between the dongle and the recorder
        ("SerialNumberDongle", c_uint),         # The Bluetooth Dongle serial number.
    ]

#---
# TMSiDongleList
#---
class TMSiDongleList(Structure):
    _pack_=1
    _fields_ = [
        ("TMSiDongleID", c_ushort),         # Unique ID to identify a bluetooth-dongle, used for pairing and connect purposes.
        ("SerialNumber", c_uint),           # The bluetooth-dongle serial number.
    ]

#---
# TMSiDevInfoReport
#---
class TMSiDevInfoReport(Structure):
    _pack_=1
    _fields_ = [
        ("SerialNumber", c_uint),                   # The Data Recorder serial number.
        ("DevID", c_ushort),                        # The device ID.
        ("APIVersion", c_ushort),                   # The current Front-End API version 0xUU.LL
        ("APIVersionString", c_char * 18),          # The current Front-End API version string
        ("FWMCUVersion", c_ushort),                 # The current firmware version 0xUU.LL
        ("FWMCUVersionString", c_char * 18),        # The current firmware version string
        ("FWBTPCVersion", c_ushort),                # The current firmware version 0xUU.LL of the PC Bluetooth module
        ("FWBTPCVersionString", c_char * 18),       # The current firmware version string of the PC Bluetooth module
        ("FWBTSensorVersion", c_ushort),            # The current firmware version 0xUU.LL of the Sensor Bluetooth module
        ("FWBTSensorVersionString", c_char * 18),   # The current firmware version string of the Sensor Bluetooth module
        ("NrOfHWChannels", c_ushort),               # Total number of hardware channels (ExG, BIP, AUX).
        ("NrOfChannels", c_ushort),                 # Total number of hardware + software channels.
        ("NrOfImpChannels", c_ushort),              # Total number of impedance channels.
        ("NrOfCyclingStates", c_ushort),            # Total number of cycling states.
        ("DeviceName", c_char * 18),                # Full device name 17 char string (zero terminated).
    ]


#---
# TMSiDevPowerStatus
#---
class TMSiDevPowerStatus(Structure):
    _pack_=1
    _fields_ = [
        ("BatTemp", c_short),               # Battery temperature.
        ("BatVoltage", c_short),            # Battery voltage in mV.
        ("BatRemainingCapacity", c_short),  # Available battery  capacity in mAh.
        ("BatFullChargeCapacity", c_short), # Max battery capacity in mAh.
        ("BatAverageCurrent", c_short),     # Discharge current of the battery in mA.
        ("BatTimeToEmpty", c_short),        # Estimated remaining min. before empty.
        ("BatStateOfCharge", c_short),      # Estimated capacity in %.
        ("BatStateOfHealth", c_short),      # Estimated battery health in %.
        ("BatCycleCount", c_short),         # Battery charge cycles.
        ("ChargeStatus", c_short),          # Battery charge cycles.
        ("ExternalPowerState", c_short),    # N.A., Connected
    ]


#---
# TMSiTime
#---
class TMSiTime(Structure):
    _pack_=1
    _fields_ = [
        ("Seconds", c_short),       # Time seconds.
        ("Minutes", c_short),       # Time minutes.
        ("Hours", c_short),         # Time Hours.
        ("DayOfMonth", c_short),    # Time Day of month.
        ("Month", c_short),         # Time month.
        ("Year", c_short),          # Years Since 1900.
        ("WeekDay", c_short),       # Day since Sunday.
        ("YearDay", c_short),       # Day since January 1st.
    ]

#---
# TMSiInterfaceStatus
#---
class TMSiInterfaceStatus(Structure):
    _pack_=1
    _fields_ = [
        ("InterfaceControl",  c_byte),    # See <TMSiControlType>
        ("InterfaceStatus",  c_byte),     # See <TMSiInterfaceControlStatusType>
    ]

#---
# TMSiDevSamplingCfg
#---
class TMSiDevSamplingCfg(Structure):
    _pack_=1
    _fields_ = [
        ("BaseSampleRate",  c_ushort),    # The fs used for this config, 1024 or 1000.
        ("ImpedanceLimit",  c_short),     # The impedance limit for signal quality.
        ("LiveImpedance",   c_byte),      # <ControlDisabled> = No Imp during measurement reference, <ControlEnabled> = LiveIMp.
    ]


#---
# TMSiDevChName
#---
class TMSiDevChName(Structure):
    _pack_=1
    _fields_ = [
        ("ChanIdx",  c_ushort),         # Which channel, in the channel-array, is this configure for.
        ("ChanName", c_char * 10),      # Default name 9 char zero
    ]

#---
# TMSiDevAltChName
#---
class TMSiDevAltChName(Structure):
    _pack_=1
    _fields_ = [
        ("ChanIdx",  c_ushort),         # Which channel, in the channel-array, is this configure for.
        ("AltChanName", c_char * 10),   # User configurable name 9 char zero
    ]


#---
# TMSiDevChanRef
#---
class TMSiDevChanRef(Structure):
    _pack_=1
    _fields_ = [
        ("ChanIdx",  c_ushort),         # Which channel, in the channel-array, is this configure for.
        ("ChanRefStatus",  c_byte),     # -1 Channel disabled, 0 Channel not in reference, 1 Channel in reference
    ]

#---
# TMSiSampleMetadataHeader
#---
class TMSiSampleMetadataHeader(Structure):
    _pack_=1
    _fields_ = [
        ("BaseFS",  c_ushort),              # The sample rate in this config
        ("NrOfChannels",  c_ushort),        # The nr of Channels in this config
        ("NrOfCyclingStates",  c_ushort),   # The nr of cycling state items in this config
        ("Interface",  c_byte),             # The interface for "streaming"
    ]


#---
# TMSiChannelMetadata
#---
class TMSiChannelMetadata(Structure):
    _pack_=1
    _fields_ = [
        ("ChanIdx", c_ushort),          # The channel index number. (starts at 0 for channel 1).
        ("ChannelType", c_ushort),      # 0=Unknown, 1=UNI, 2=BIP, 3=AUX, 4=DIGRAW/Sensor, 5=STATUS, 6=COUNTER, 7=IMP, 8=Cycl
        ("ChannelFormat", c_ushort),    # 0x11xx Float, 0x00xx Unsigned xx bits, 0x01xx signed xx bits
        ("ChanDivider", c_short),       # -1 disabled, else BaseSampleRateHz >> ChanDivider.
        ("ImpDivider", c_short),        # -1 disabled, else BaseSampleRate>>ImpDivider
        ("ChannelBandWidth", c_int),    # Bandwidth (in MB/s) required for transfer from DR to DS, used by bandwidth manager in application software.
        ("Unitconva", c_float),         # Unit = a*Bits + b
        ("Unitconvb", c_float),         #
        ("Exp", c_short),               # Exponent, 3= kilo,  -6 = micro etc.
        ("UnitName", c_char * 10),      # Channel Unit, 9 char zero terminated.
        ("ChanName", c_char * 10),      # Default channel name 9 char zero terminated.
        ("AltChanName", c_char * 10),   # User configurable name 9 char zero terminated.
    ]


#---
# TMSiCyclingStateMetadata
#---
class TMSiCyclingStateMetadata(Structure):
    _pack_=1
    _fields_ = [
        ("CyclingIdx", c_ushort),       # The cycling index number. (starts at 0).
        ("ST1Format", c_ushort),        # 0x11xx Float, 0x00xx Unsigned xx bits, 0x01xx signed xx bits
        ("ST2Format", c_ushort),        # 0x11xx Float, 0x00xx Unsigned xx bits, 0x01xx signed xx bits
        ("ST1Unitconva", c_float),      # Unit = a*Bits + b
        ("ST1Unitconvb", c_float),      #
        ("ST2Unitconva", c_float),      # Unit = a*Bits + b
        ("ST2Unitconvb", c_float),      #
        ("ST1Exp", c_short),            # Exponent, 3= kilo,  -6 = micro etc.
        ("ST2Exp", c_short),            # Exponent, 3= kilo,  -6 = micro etc.
        ("ST1UnitName", c_char * 10),   # Channel Unit, 9 char zero terminated.
        ("ST1ChanName", c_char * 10),   # Default channel name 9 char zero terminated.
        ("ST1AltChanName", c_char * 10),# User configurable name 9 char zero terminated.
        ("ST2UnitName", c_char * 10),   # Channel Unit, 9 char zero terminated.
        ("ST2ChanName", c_char * 10),   # Default channel name 9 char zero terminated.
        ("ST2AltChanName", c_char * 10),# User configurable name 9 char zero terminated.
    ]

#---
# TMSiImpedanceMetadata
#---
class TMSiImpedanceMetadata(Structure):
    _pack_=1
    _fields_ = [
        ("ChanIdx", c_ushort),              # The channel index number. (starts at 0 for channel 1)..    
        ("ChanName", c_char * 10),          # Default channel name 9 char zero terminated.
        ("ImpedanceReUnit", c_char * 10),   # Impedance Unit name , real part, 9 char zero terminated.
        ("ImpedanceImUnit", c_char * 10),   # Impedance Unit name, imaginary part, 9 char zero terminated.
    ]
    
#---
# TMSiImpedanceSample
#---
class TMSiImpedanceSample(Structure):
    _pack_=1
    _fields_ = [
        ("ChanIdx", c_ushort),      # The channel index number. (starts at 0 for channel 1)..    
        ("ImpedanceRe", c_short),   # Real-part of the impedance sample value.
        ("ImpedanceIm", c_short),   # Imaginary-part of the impedance sample value.
    ]    


#---
# TMSiDevSampleRequest
#---
class TMSiDevSampleRequest(Structure):
    _pack_=1
    _fields_ = [
        ("StartStop", c_byte),          # Flag to start <StartSampling> and stop <StopSampling>
        ("DisableLiveImp", c_byte),     # Disable the Live-Impedance for this measurement.
        ("DisableAvrRefCalc", c_byte),  # Disable average reference calculation for now.
    ]


#---
# TMSiDevImpedanceRequest
#---
class TMSiDevImpedanceRequest(Structure):
    _pack_=1
    _fields_ = [
        ("StartStop", c_byte),          # Flag to start <StartImpedance> and stop <StopImpedance>
    ]

#---
# TMSiDevCardRecCfg
#---
class TMSiDevCardRecCfg(Structure):
    _pack_=1
    _fields_ = [
        ("ProtoVer", c_short),              # Version of the current spec used.
        ("FileType", c_short),              # Type of file set by device.
        ("StartControl", c_short),          # Configuration how to start the ambulant recording.
        ("EndControl", c_int),              # Configuration how to stop the amplulant recording.
        ("StorageStatus", c_int),           # Status of the internal storage.
        ("InitIdentifier", c_int),          # Identifier can be used by the application.
        ("PrefixFileName", c_char * 16),    # Prefix for the final recording filename.
        ("StartTime", TMSiTime),            # The start time for the recording.
        ("StopTime", TMSiTime),             # The stop time for the recording.
        ("PreImp", c_short),                # Pre measurement impedance 0=no, 1=yes.
        ("PreImpSec", c_short),             # Amount of seconds for impedance.
        ("UserString1", c_char * 64),       # Freeformat string, can be set by application.
        ("UserString2", c_char * 64),       # Freeformat string, can be set by application.
    ]


#---
# TMSiDevCardStatus
#---
class TMSiDevCardStatus(Structure):
    _pack_=1
    _fields_ = [
        ("NrOfRecordings", c_ushort),   # Number of available card recordings.
        ("TotalSpace", c_uint),         # Total space in MB
        ("AvailableSpace", c_uint),     # Free space in MB
    ]

#---
# TMSiDevCardFileInfo
#---
class TMSiDevCardFileInfo(Structure):
    _pack_=1
    _fields_ = [
        ("RecFileID", c_ushort),        # Identifier for this file.
        ("RecFileName", c_char * 32),   # Filename
        ("StartTime", TMSiTime),        # StartTime of this recording.
        ("StopTime", TMSiTime),         # StopTime of this recording.
    ]


#---
# TMSiFileMetadataHeader
#---
class TMSiFileMetadataHeader(Structure):
    _pack_=1
    _fields_ = [
        ("BaseFS", c_ushort),                   #
        ("NrOfChannels", c_ushort),             # The nr of Channels in this config
        ("NrOfCyclingStates", c_ushort),        # The nr of cycling states in this config
        ("NrOfImpReportChannels", c_ushort),    # The nr of impedance report channels.
        ("Interface", c_byte),                  #
    ]

#---
# TMSiDevCardFileDetails,
#---
class TMSiDevCardFileDetails(Structure):
    _pack_=1
    _fields_ = [
        ("RecFileID", c_ushort),
        ("StorageStatus", c_int),
        ("NumberOfSamples", c_uint),
        ("RecFileName", c_char * 32),
        ("StartTime", TMSiTime),
        ("StopTime", TMSiTime),
        ("ImpAvailable", c_short),
        ("UserString1", c_char * 64),
        ("UserString2", c_char * 64),
    ]



#---
# TMSiDevImpReportMetadata
#---
class TMSiDevImpReportMetadata(Structure):
    _pack_=1
    _fields_ = [
        ("NrOfImpReportChannels", c_ushort),    # The nr of impedance report channels.
        ("ImpedanceReUnit", c_char * 10),       # Impedance Unit name , real part, 9 char zero terminated.
        ("ImpedanceImUnit", c_char * 10),       # Impedance Unit name, imaginary part, 9 char zero terminated.        
    ]
    
#---
# TMSiDevImpReport
#---
class TMSiDevImpReport(Structure):
    _pack_=1
    _fields_ = [
        ("ChanIdx", c_ushort),      # The channel for which this impedance value is.
        ("ChanName", c_char * 10),          # Default channel name 9 char zero terminated.
        ("ImpedanceRe", c_short),   # Real-part of the impedance sample value.
        ("ImpedanceIm", c_short),   # Imaginary-part of the impedance sample value.        
    ]


#---
# TMSiDevSetCardFileReq
#---
class TMSiDevSetCardFileReq(Structure):
    _pack_=1
    _fields_ = [
        ("StartStop", c_ushort),  # The channel for which this impedance value is.
        ("RecFileID", c_ushort),   # The actual impedance value for this channel.
        ("StartCounter", c_uint),   # The actual impedance value for this channel.
        ("NumberOfSamples", c_uint),   # The actual impedance value for this channel.
    ]


#---
# TMSiDevChCal
#---
class TMSiDevChCal(Structure):
    _pack_=1
    _fields_ = [
        ("ChanIdx", c_uint),            # Which channel is this configure for.
        ("ChanGainCorr", c_float),      # A float value for the Gain calibration.
        ("ChanOffsetCorr", c_float),    # A float value for the Offset calibration.
    ]



#---
# TMSiDevProductConfig
#---
class TMSiDevProductConfig(Structure):
    _pack_=1
    _fields_ = [
        ("DRSerialNumber", c_uint),     # The DR serial number.
        ("DRDevID", c_ushort),          # DR Device ID
        ("NrOfHWChannels", c_ushort),   # total nr of UNI, Bip, Aux channels.
        ("NrOfChannels", c_ushort),     # Total number of channels.
    ]


#---
# TMSiDevProductChCfg
#---
class TMSiDevProductChCfg(Structure):
    _pack_=1
    _fields_ = [
        ("ChannelType", c_ushort),      # 0=Unknown, 1=UNI, 2=BIP, 3=AUX, 4=DIGRAW/Sensor,5=DIGSTAT, 6=SAW.
        ("ChannelFormat", c_ushort),    # 0x00xx Usigned xx bits, 0x01xx signed xx bits
        ("Unitconva", c_float),         # Unit = a*Bits + b used for bits -> unit, IEEE 754
        ("Unitconvb", c_float),
        ("Exp", c_short),               # Exponent, 3= kilo,  -6 = micro etc.
        ("UnitName", c_char * 10),      # Channel Unit, 9 char zero terminated.
        ("DefChanName", c_char * 10),   # Default channel name 9 char zero terminated.
    ]


#---
# TMSiEvent
#---
class TMSiEvent(Structure):
    _pack_=1
    _fields_ = [
        ("EventID", c_uint),        # See <TMSiEventIDType>
        ("Code", c_uint),           # See <TMSiEventCodeType>
        ("TMSiDeviceID", c_ushort), # Unique ID to identify device, retrieved with TMSi_GetDeviceList().
        ("TMSiDongleID", c_ushort), # Unique ID to identify a bluetooth-dongle, retrieved with TMSi_GetDongleList().
    ]


#---
# TMSiDeviceDriverVersionInfo
#---
class TMSiDeviceDriverVersionInfo(Structure):
    _pack_=1
    _fields_ = [
        ("DllVersion", c_uint),                 # The current dll version 0xUU.LL
        ("DllVersionString", c_char * 18),      # The current dll API version string
        ("LibUsbVersion", c_uint),              # The current libusb version 0xUU.LL
        ("LibUsbVersionString", c_char * 18),   # The current libusb version string
    ]