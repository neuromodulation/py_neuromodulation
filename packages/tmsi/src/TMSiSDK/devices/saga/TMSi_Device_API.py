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
 * @file ${TMSi_Device_API.py} 
 * @brief SAGA Device API protocol definitions VERSION 1.4
 *
 */


'''

from ctypes import *
from sys import platform
import os
import pdb
from array import *
from ...error import TMSiError, TMSiErrorCode

from enum import Enum, unique

TMSiDeviceRetVal = c_uint

DeviceHandle = c_void_p
TMSiDeviceHandle = DeviceHandle(0)
SagaDllAvailable = False
SagaDllLocked = True


if platform == "linux" or platform == "linux2":

    so_name = "libTMSiSagaDeviceLib.so"
    soabspath = os.path.sep + os.path.join('usr', 'lib', so_name)
    dlclose_func = cdll.LoadLibrary('').dlclose
    dlclose_func.argtypes = [c_void_p]

    try:
        CDLL("librt.so.1",  RTLD_GLOBAL)
        SagaSDK = CDLL(soabspath,  RTLD_GLOBAL)
        sdk_handle = SagaSDK._handle
        print("Successfully loaded SAGA device library, handle: " + hex(sdk_handle) )
    except Exception as e:
        print(e)
elif platform == "win32": # Windows
    search_path = "C:/Program files/TMSi/Saga"
    name = "TMSiSagaDeviceLib.dll"
    result = os.path.join(search_path, name)
    so_name = os.path.abspath(result)
    if os.path.exists(so_name):
        print("{} available.".format(so_name))
        SagaDllAvailable = True
    try:
        SagaSDK = CDLL(so_name)
        SagaDllLocked = False
        sdk_handle = SagaSDK._handle
        print("Successfully loaded SAGA device library, handle: " + hex(sdk_handle) )
    except Exception as e:
        if SagaDllAvailable:
            print("{} already in use.".format(so_name))
else:
    print("Unsupported platform")

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
	TMSI_OK = 0x00000000					# All Ok positive ACK.
	TMSI_DR_CHECKSUM_ERROR = 0x01010001			# DR reported "Checksum error in received block".
	TMSI_DS_CHECKSUM_ERROR = 0x02010001			# DS reported "Checksum error in received block".
	TMSI_DR_UNKNOWN_COMMAND = 0x01010002 			# DR reported "Unknown command".
	TMSI_DS_UNKNOWN_COMMAND = 0x02010002 			# DS reported "Unknown command".
	TMSI_DR_RESPONSE_TIMEMOUT = 0x01010003 			# DR reported "Response timeout".
	TMSI_DS_RESPONSE_TIMEMOUT = 0x02010003 			# DS reported "Response timeout".
	TMSI_DR_DEVICE_BUSY = 0x01010004 			# DR reported "Device busy try again in x msec".
	TMSI_DS_DEVICE_BUSY = 0x02010004 			# DS reported "Device busy try again in x msec".
	TMSI_DR_COMMAND_NOT_SUPPORTED = 0x01010005 		# DR reported "Command not supported over current interface".
	TMSI_DS_COMMAND_NOT_SUPPORTED = 0x02010005 		# DS reported "Command not supported over current interface".
	TMSI_DR_COMMAND_NOT_POSSIBLE = 0x01010006 		# DR reported "Command not possible, device is recording".
	TMSI_DR_DEVICE_NOT_AVAILABLE = 0x01010007 		# DR reported "Device not available".
	TMSI_DS_DEVICE_NOT_AVAILABLE = 0x02010007 		# DS reported "Device not available".
	TMSI_DS_INTERFACE_NOT_AVAILABLE = 0x02010008 		# DS reported "Interface not available".
	TMSI_DS_COMMAND_NOT_ALLOWED = 0x02010009 		# DS reported "Command not allowed in current mode".
	TMSI_DS_PROCESSING_ERROR = 0x0201000A 			# DS reported "Processing error".
	TMSI_DS_UNKNOWN_INTERNAL_ERROR = 0x0201000B 		# DS reported "Unknown internal error".
	TMSI_DR_COMMAND_NOT_SUPPORTED_BY_CHANNEL = 0x01030001 	# DR reported "Command not supported by Channel".
	TMSI_DR_AMBREC_ILLEGAL_START_CTRL = 0x01030002 		# DR reported "Illegal start control for ambulant recording".

	# Additional defines below for DS error types.
	TMSI_DS_PACKET_LENGTH_ERROR = 0x0201000C 		# DS reports that data request does not fit with one Device Api Packet
	TMSI_DS_DEVICE_ALREADY_OPEN = 0x0201000D 		# DS reports that DR is already opened.

	# Additional defines below for DLL error types.
	TMSI_DLL_NOT_IMPLEMENTED = 0x03001000 			# DLL Function is declared, but not yet implemented
	TMSI_DLL_INVALID_PARAM = 0x03001001 			# DLL Function called with invalid parameters
	TMSI_DLL_CHECKSUM_ERROR = 0x03001002
	TMSI_DLL_ETH_HEADER_ERROR = 0x03001003
	TMSI_DLL_INTERNAL_ERROR = 0x03001004 			# DLL Function failed because an underlying process failed
	TMSI_DLL_BUFFER_ERROR = 0x03001005 			# DLL Function called with a too small buffer
	TMSI_DLL_INVALID_HANDLE = 0x03001006 			# DLL Function called with a Handle that's not assigned to a device
	TMSI_DLL_INTF_OPEN_ERROR = 0x03002000
	TMSI_DLL_INTF_CLOSE_ERROR = 0x03002001
	TMSI_DLL_INTF_SEND_ERROR = 0x03002002
	TMSI_DLL_INTF_RECV_ERROR = 0x03002003
	TMSI_DLL_INTF_RECV_TIMEOUT = 0x03002004
	TMSI_DLL_LOST_CONNECTION = 0x03002005 			# Lost connection to DS, USB / Ethernet disconnect.


# Communication interface used
# 0 = Unknown, 1=USB 2=Nework, 3=WiFi, 4=Electrical, 5=Optical, 6=Bluetooth.
@unique
class TMSiInterface(Enum):
	IF_TYPE_UNKNOWN = 0
	IF_TYPE_USB = 1
	IF_TYPE_NETWORK = 2
	IF_TYPE_WIFI = 3
	IF_TYPE_ELECTRICAL = 4
	IF_TYPE_OPTICAL = 5
	IF_TYPE_BLUETOOTH = 6


# Protocol block definitions

#---
# TMSiDevList
#---
class TMSiDevList(Structure):
    _pack_=2
    _fields_ = [
        ("TMSiDeviceID", c_ushort),     # Unique ID to identify device, used to open device.
        ("DSSerialNr", c_uint),         # The DS serial number.
        ("DRAvailable", c_ushort),      # defined as 0 = DR_Offline, 1 = DR_Online
        ("DRSerialNr", c_uint),         # The DR serial number.
    ]


#---
# TMSiDevStatReport
#---
class TMSiDevStatReport(Structure):
    _pack_=2
    _fields_ = [
        ("DSSerialNr", c_uint),         # The DS serial number.
        ("DRSerialNr", c_uint),         # The DR serial number.
        ("DSInterface", c_ushort),      # Communication interface on DS used, 0 = Unknown, 1=USB, 2=Network, 3=WiFi, 4=Electrical, 5=Optical, 6=Bluetooth.
        ("DRInterface", c_ushort),      # Communication interface on DR used, 0 = Unknown, 1=USB  2=Network, 3=WiFi, 4=Electrical, 5=Optical, 6=Bluetooth.
        ("DSDevAPIVersion", c_ushort),  # Current Device-API version used by DS, V00.00
        ("DRAvailable", c_ushort),      # defined as 0 = FE_Offline, 1 = FE_Online
        ("NrOfBatteries", c_ushort),    # Nr of batteries, indicates nr of TMSiBatReport.
        ("NrOfChannels", c_ushort),     # Total number of channels for the device.
    ]


#---
# TMSiDevFullStatReport
#---
class TMSiDevFullStatReport(Structure):
    _pack_=2
    _fields_ = [
        ("DSSerialNr", c_uint),         # The DS serial number.
        ("DRSerialNr", c_uint),         # The DR serial number.
        ("DRWiFiPaired", c_uint),       # The serial number of paired DR over WiFi.
        ("KeepAliveTimeout", c_short),  # DR Idle powerdown in sec. -1 disabled.
        ("PowerState", c_ushort),       # 0 = Unknown, 1 = External, 2 = Battery.
        ("DSTemp", c_short),            # DS temperature in degree Celsius.
        ("DRTemp", c_short),            # DR temperature in degree Celsius.
    ]


#---
# TMSiDevBatReport
#---
class TMSiDevBatReport(Structure):
    _pack_=2
    _fields_ = [
        ("BatID", c_short),                 # 0 unknown, 1, 2 etc.
        ("BatTemp", c_short),               # Battery temperature.
        ("BatVoltage", c_short),            # Battery voltage in mV.
        ("BatRemainingCapacity", c_short),  # Available battery capacity in mAh.
        ("BatFullChargeCapacity", c_short), # Max battery capacity in mAh.
        ("BatAverageCurrent", c_short),     # Current going in or out of the battery in mA.
        ("BatTimeToEmpty", c_short),        # Estimated remaining minutes before empty in min.
        ("BatStateOfCharge", c_short),      # Estimated capacity in %.
        ("BatStateOfHealth", c_short),      # Estimated battery health in %.
        ("BatCycleCount", c_short),         # Battery charge cycles.
    ]


#---
# TMSiTime
#---
class TMSiTime(Structure):
    _pack_=2
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
# TMSiDevStorageReport
#---
class TMSiDevStorageReport(Structure):
    _pack_=2
    _fields_ = [
        ("TotalSizeMB", c_uint),    # Total storage in MByte.
        ("UsedSizeMB", c_uint),     # Available storage in MByte.
    ]


#---
# TMSiDevGetConfig
#---
class TMSiDevGetConfig(Structure):
    _pack_=2
    _fields_ = [
        ("DRSerialNumber", c_uint),         # The DR serial number.
        ("DRDevID", c_ushort),              # The DR Device ID.
        ("NrOfHWChannels", c_ushort),       # Total nr of hardware channels (UNI, Bip, Aux).
        ("NrOfChannels", c_ushort),         # Total number of hardware + software channels.
        ("NrOfSensors", c_ushort),          # Total supported sensor interfaces.
        ("BaseSampleRateHz", c_ushort),     # Current base samplerate
        ("AltBaseSampleRateHz", c_ushort),  # 4096 / 4000 depends on BaseSampleRateHz.
        ("ConfiguredInterface", c_ushort),  # Communication interface on DR used, 0 = Unknown, 1=USB  2=Network, 3=WiFi, 4=Electrical, 5=Optical, 6=Bluetooth.
        ("InterFaceBandWidth", c_int),      # Data bandwidth in MB/s for current interface.
        ("TriggersEnabled", c_short),       # 0= disabled 1 = External triggers enabled.
        ("RefMethod", c_short),             # 0= Common reference, 1=average reference.
        ("AutoRefMethod", c_short),         # 0= fixed method, 1= if average -> common reference.
        ("DRSyncOutDiv", c_short),          # BaseSampleRate/SyncOutDiv,  -1 = markerbut.
        ("DRSyncOutDutyCycl", c_short),     # SyncOutv dutycycle.
        ("DSSyncOutDiv", c_short),          # BaseSampleRate/SyncOutDiv,  -1 = markerbut.
        ("DSSyncOutDutyCycl", c_short),     # SyncOutv dutycycle, relative to DR BaseFreq
        ("RepairLogging", c_short),         # 0 Disabled, 1 = BackupLogging enabled AmbRecording is disabled.
        ("AmbRecording", c_short),          # 0 Disabled, 1 = Ambulant Configured/Enabled
        ("AvailableRecordings", c_short),   # Currently stored recordings on device.
        ("DeviceName", c_char * 18),        # Full dev name 17 char string (zero terminated).
    ]


#---
# TMSIDevChDesc
#---
class TMSiDevChDesc(Structure):
    _pack_=2
    _fields_ = [
        ("ChannelType", c_ushort),      # 0=Unknown, 1=UNI, 2=BIP, 3=AUX, 4=DIGRAW/Sensor,5=DIGSTAT, 6=SAW.
        ("ChannelFormat", c_ushort),    # 0x00xx Usigned xx bits, 0x01xx signed xx bits
        ("ChanDivider", c_short),       # -1 disabled, else BaseSampleRateHz >> ChanDivider.
        ("ImpDivider", c_short),        # -1 disabled, else BaseSampleRate>>ImpDivider
        ("ChannelBandWidth", c_int),    # Bandwidth (in MB/s) required for transfer from DR to DS, used by bandwidth manager in application software.
        ("Exp", c_short),               # Exponent, 3= kilo,  -6 = micro etc.
        ("UnitName", c_char * 10),      # Channel Unit, 9 char zero terminated.
        ("DefChanName", c_char * 10),   # Default channel name 9 char zero terminated.
        ("AltChanName", c_char * 10),   # User configurable name 9 char zero terminated.
    ]


#---
# The basic sensor metadata header describing the type of sensor
#---
class SensorDataHeader(Structure):
    _pack_=2
    _fields_ = [
	    ("ManufacturerID", c_ushort),       # Who makes this accessory.
    	("Serialnr", c_uint),               # Serial number.
    	("ProductIdentifier", c_ulonglong), # Together with (AI) and serial number forming UDI
    	("ChannelCount", c_ubyte),          # Indicates the number of channel structs
    	("AdditionalStructs", c_ubyte),     # Indicates the number of additional Structs
    ]


#---
# Sensor Structure [StructID 0x0000] 32 bytes
#---
class SensorDefaultChannel(Structure):
    _pack_=2
    _fields_ = [
    	("StructID", c_ushort),         # ID (= 0x0000)of the channel struct according to XML-file
    	("ChannelName", c_char * 10),   # Zero terminated string for the channel name.
    	("UnitName", c_char * 10),      # Zero terminated string for the Unit string.
    	("Exponent", c_short),          # Exponent for the unit, e.g. milli = -3 this gives for a UnitName V a result mV.
    	("Gain", c_float),              # Value to convert the sensor value to the correct unit value.
    	("Offset", c_float),            # Offset for this channel.
    ]


#---
# Sensor Structure [StructID 0xFFFF] 2 bytes
#---
class SensorDummyChannel(Structure):
    _pack_=2
    _fields_ = [
	    ("StructID", c_ushort), # ID (= 0xFFFF) of the dummy channel struct
    ]

#---
# Sensor Structure [StructID 0x03FE] 74 bytes
#---
class TMSiContact(Structure):
    _pack_=2
    _fields_ = [
    	("StructID", c_ushort),         #  ID (= 0x3FE) of the TMSi Contact struct according to XML-file
    	("CompanyName", c_char * 10),   # Zero terminated string for the Company name.
    	("WWW", c_char * 20),           # Zero terminated string for web URL.
    	("Email", c_char * 30),         # Zero terminated string for support email.
        ("Phone", c_char * 16),         # Zero terminated string for telephone number.
    ]


#---
# TMSiDevSetConfig
#---
class TMSiDevSetConfig(Structure):
    _pack_=2
    _fields_ = [
        ("DRSerialNumber", c_uint),             # The DR serial number.
        ("NrOfChannels", c_ushort),             # total nr of channels in this configuration.
        ("SetBaseSampleRateHz", c_ushort),      # New base samplerate for DR.
        ("SetConfiguredInterface", c_ushort),   # Comm interface on DR to configure, 0 = Unknown, 1=USB  2=Network, 3=WiFi, 4=Electrical,5=Optical, 6=Bluetooth.
        ("SetTriggers", c_short),               # 0= Disabled, 1= external triggers enabled.
        ("SetRefMethod", c_short),              # 0= Common reference, 1=average reference.
        ("SetAutoRefMethod", c_short),          # 0= fixed method, 1= if average -> common.
        ("SetDRSyncOutDiv", c_short),           # SetBaseSampleRateHz/SyncOutDiv, -1 = markerbut.
        ("DRSyncOutDutyCycl", c_short),         # Set DR Sync dutycycle.
        ("SetRepairLogging", c_short),          # 0 Disabled, 1 = BackupLogging enabled, Ambulatory recording is disabled!
        ("PerformFactoryReset", c_short),       # Set device to defaults, all other options in this config are ignored by device.
        ("StoreAsDefault", c_short),            # Set the current configuration as default startup configuration.
        ("WebIfCtrl", c_ushort),                # Status of DS WebIF stop = 0, start =1
        ("PinKey", c_byte * 4),                 # Pincode to use for pairing procedure.
    ]


#---
# TMSiDevSetChCfg
#---
class TMSiDevSetChCfg(Structure):
    _pack_=2
    _fields_ = [
        ("ChanNr", c_ushort),           # Which channel is this configure for.
        ("ChanDivider", c_short),       # -1 disabled, else SetBaseSampleRateHz>>ChanDivider.
        ("AltChanName", c_byte * 10),   # User configurable name 9 char zero terminated.
    ]


#---
# TMSiDevGetSens
#---
class TMSiDevGetSens(Structure):
    _pack_=2
    _fields_ = [
        ("ChanNr", c_ushort),               # Channel where sensor is connected to.
        ("IOMode", c_short),                # -1 Disabled, 0=Nonin, 1=SPI, 2=UART, 20=Analog
        ("SensorID", c_short),              # ID of connected Sensor to this channel
        ("SensorMetaData", c_byte * 128),   # Additional raw data from sensor.
    ]


#---
# TMSiSetDevSens
#---
class TMSiSetDevSens(Structure):
    _pack_=2
    _fields_ = [
        ("ChanNr", c_ushort),   # Channel of connected sensor
        ("IOMode", c_short),    # Channel Sensor communication method: -1 Disabled, 0=Nonin, 1=SPI, 2=UART, 20=Analog
    ]


#---
# TMSiDevSampleReq
#---
class TMSiDevSampleReq(Structure):
    _pack_=2
    _fields_ = [
        ("SetSamplingMode", c_ushort),      # flag to start and stop, see SampleControlType commands.
        ("DisableAutoswitch", c_ushort),    # Ignore the Refmode autoswitch for now.
        ("DisableRepairLogging", c_ushort), # Ignore the Repairlogging for now.
        ("DisableAvrRefCalc", c_ushort),    # Disable average ref. calculation for now.
    ]


#---
# SampleControl, enum for SetSamplingMode values.
#---
@unique
class SampleControl(Enum):
    STOPSamplingDevice = 0
    STARTSamplingDevice = 1
    STOPWiFiStream = 2


#---
# TMSiDevImpReq
#---
class TMSiDevImpReq(Structure):
    _pack_=2
    _fields_ = [
        ("SetImpedanceMode", c_ushort), # flag to start and stop, bitmask defined as ImpedanceControlType can be used.
    ]


#---
# ImpedanceControl, enum for SetImpedanceMode values.
#---
@unique
class ImpedanceControl(Enum):
    ImpedanceStop = 0
    ImpedanceStart = 1

#---
# TMSiDevRecList
#---
class TMSiDevRecList(Structure):
    _pack_=2
    _fields_ = [
        ("RecFileID", c_ushort),        # Identifier for this file.
        ("RecFileName", c_char * 32),   # Filename
        ("StartTime", TMSiTime),        # StartTime of this recording.
        ("StopTime", TMSiTime),         # StopTime of this recording.
    ]


#---
# TMSiDevRecDetails,
#---
class TMSiDevRecDetails(Structure):
    _pack_=2
    _fields_ = [
        ("StructID", c_short),
        ("ProtoVer", c_short),
        ("RecFileType", c_short),
        ("RecFileID", c_short),
        ("StorageStatus", c_int),
        ("NoOfSamples", c_int),
        ("RecFileName", c_char * 32),
        ("StartTime", TMSiTime),
        ("StopTime", TMSiTime),
        ("ImpAvailable", c_short),
        ("PatientID", c_char * 128),
        ("UserString1", c_char * 64),
        ("UserString2", c_char * 64),
    ]


#---
# TMSiDevImpReport
#---
class TMSiDevImpReport(Structure):
    _pack_=2
    _fields_ = [
        ("ChanNr", c_ushort),   # The channel for which this impedance value is.
        ("Impedance", c_float), # The actual impedance value for this channel.
    ]


#---
# TMSiDevRecCfg
#---
class TMSiDevRecCfg(Structure):
    _pack_=2
    _fields_ = [
        ("ProtoVer", c_short),              # Version of the current spec used.
        ("FileType", c_short),              # Type of file set by device.
        ("StartControl", c_short),          # Configuration how to start the ambulant recording.
        ("EndControl", c_int),              # Configuration how to stop the amplulant recording.
        ("StorageStatus", c_int),           # Status of the internal storage.
        ("InitIdentifier", c_int),          # Identifier can be used by the application.
        ("PrefixFileName", c_byte * 16),    # Prefix for the final recording filename.
        ("StartTime", TMSiTime),            # The start time for the recording.
        ("StopTime", TMSiTime),             # The stop time for the recording.
        ("IntervalSeconds", c_short),       # Recuring start time seconds.
        ("IntervalMinutes", c_short),       # Recuring start time minutes.
        ("IntervalHours", c_short),         # Recuring start time hours.
        ("IntervalDays", c_short),          # Recuring start time days.
        ("AlarmTimeCount", c_int),          # Amount of recurring cycles.
        ("PreImp", c_short),                # Pre measurement impedance 0=no, 1=yes.
        ("PreImpSec", c_short),             # Amount of seconds for impedance.
        ("PatientID", c_byte * 128),        # Freeformat string, can be set by application.
        ("UserString1", c_byte * 64),       # Freeformat string, can be set by application.
        ("UserString2", c_byte * 64),       # Freeformat string, can be set by application.
    ]


#---
# TMSiDevRepairReq
#---
class TMSiDevRepairReq(Structure):
    _pack_=2
    _fields_ = [
        ("SampleStartCntr", c_uint),    # Sample Saw counter Start.
        ("NROfSampleSets", c_uint),     # Number of sets to retrieve.
    ]


#---
# TMSiDevChCal
#---
class TMSiDevChCal(Structure):
    _pack_=2
    _fields_ = [
        ("ChanNr", c_uint),             # Which channel is this configure for.
        ("ChanGainCorr", c_float),      # A float value for the Gain calibration.
        ("ChanOffsetCorr", c_float),    # A float value for the Offset calibration.
    ]


#---
# TMSiDevGetDiagStat
#---
class TMSiDevGetDiagStat(Structure):
    _pack_=2
    _fields_ = [
        ("DRHealthState", c_ushort),    # Current state of device since last boot, 0 = OK, 1 = Error.
        ("DRErrors", c_short),          # Nr of Errors logged since logging was started, -1 logging disabled.
        ("DRLogSize", c_uint),          # DR Log size in Bytes.
        ("DSHealthState", c_ushort),    # Current state of device since last boot, 0 = OK, 1 = Error.
        ("DSErrors", c_short),          # Nr of Errors logged since logging was started, -1 logging disabled.
        ("DSLogSize", c_uint),          # DS Log size in Bytes.
    ]


#---
# TMSiDevSetDiagStat
#---
class TMSiDevSetDiagStat(Structure):
    _pack_=2
    _fields_ = [
        ("DRLoggingState", c_ushort),   # Set logging state of device, 0=disable, 1= enable.
        ("DRResetLog", c_ushort),       # Reset the logfile, old loggings are erased.
        ("DSLoggingState", c_ushort),   # Set logging state of device, 0=disable, 1= enable.
        ("DSResetLog", c_ushort),       # Reset the logfile, old loggings are erased.
    ]


#---
# FWStatus, enum for FWStatus values.
#---
@unique
class FirmwareStatus(Enum):
    FWStatus_Unknown = -1
    All_OK = 0
    Upgrading = 1
    Verify_OK = 2
    Verify_Fail = 3


#---
# TMSiDevFWStatusReport
#---
class TMSiDevFWStatusReport(Structure):
    _pack_=2
    _fields_ = [
        ("FWVersion", c_short),     # -1 N.A., Vaa.bb  -> 0xaabb
        ("AppVersion", c_short),    # -1 N.A., Vaa.bb  -> 0xaabb
        ("FWStatus", c_int),        # -1 Unknown, All_OK, Upgrading, Verify_OK, Verify_Fail,
        ("MaxPushSize", c_uint),    # max allowed block size in bytes when sending firmware to device.
    ]


#---
# TMSiFWHeaderFile
#---
class TMSiFWHeaderFile(Structure):
    _pack_=2
    _fields_ = [
        ("FWVersion", c_short),         # The Firmware version of this file.
        ("FWHardwareVersion", c_short), # The hardware version for this firmware 0xVVRR.
        ("DevID", c_short),             # The DevID for which this firmware is intended.
        ("FWSize", c_uint),             # The total size in bytes of the firmware to flash.
        ("Checksum", c_int),            # Integrity check type t.b.d.
    ]


#---
# TMSiDevProductConfig
#---
class TMSiDevProductConfig(Structure):
    _pack_=2
    _fields_ = [
        ("DRSerialNumber", c_uint),     # The DR serial number.
        ("DSSerialNumber", c_uint),     # The DS serial number.
        ("DRDevID", c_ushort),          # DR Device ID
        ("DSDevID", c_ushort),          # DS Device ID
        ("NrOfHWChannels", c_ushort),   # total nr of UNI, Bip, Aux channels.
        ("NrOfChannels", c_ushort),     # Total number of channels.
    ]


#---
# TMSiDevProductChCfg
#---
class TMSiDevProductChCfg(Structure):
    _pack_=2
    _fields_ = [
        ("ChannelType", c_ushort),      # 0=Unknown, 1=UNI, 2=BIP, 3=AUX, 4=DIGRAW/Sensor,5=DIGSTAT, 6=SAW.
        ("ChannelFormat", c_ushort),    # 0x00xx Usigned xx bits, 0x01xx signed xx bits
        ("Unitconva", c_float),         # Unit = a*Bits + b used for bits -> unit, IEEE 754
        ("Unitconvb", c_float),
        ("Exp", c_short),               # Exponent, 3= kilo,  -6 = micro etc.
        ("UnitName", c_byte * 10),      # Channel Unit, 9 char zero terminated.
        ("DefChanName", c_byte * 10),   # Default channel name 9 char zero terminated.
    ]


#---
# TMSiDevNetworkConfig
#---
class TMSiDevNetworkConfig(Structure):
    _pack_=2
    _fields_ = [
        ("NetworkMode", c_ushort),      # 0 = Network disabled, 1 = Use DHCP, 2 = Use config as below.
        ("DSIPAddress", c_byte * 16),   # Static DS IP Address.
        ("DSNetmask", c_byte * 16),     # Static DS Netmask.
        ("DSGateway", c_byte * 16),     # Static DS Gateway Address.
    ]


#---
# TMSiDevice enum used to specify the DS or DR in API calls.
#---
@unique
class TMSiDevice(Enum):
    Dev_Unknown = 0
    Dev_DS = 1
    Dev_DR = 2


#---
# FWAction enum used to specify the action to perform during FWupdate.
#---
@unique
class FWAction(Enum):
    FWAct_Unknown = 0
    FWAct_Flash_Reboot = 1
    FWAct_ABORT = 2


if SagaDllAvailable and not SagaDllLocked:
    # DLL interface

    #---
    # @details This command is used to retrieve a list of available TMSi devices
    # connected to the PC. This query is performed on the "DSInterfaceType" specified
    # by the user. All other interface types are ignored. For each device found on
    # the PC matching "DSInterfaceType" the appropriate low level command is send with
    # "DRInterfaceType" set.
    #
    # @Pre \ref No device should have been opened. Device is in Close state.
    #
    # @Post No device change.
    #
    # @depends Low level call 0x0101.
    #
    # @param[out] TMSiDeviceList  List of found devices.
    # @param[in] DSInterfaceType Interface to DS to query. 0 = Unknown,1=USB,
    # 2=Network, 3=WiFi, 4=Electrical, 5=Optical, 6=Bluetooth.
    #
    # @param[in] DRInterfaceType Interface to DR to query. 0 = Unknown,1=USB,
    # 2=Network, 3=WiFi, 4=Electrical, 5=Optical, 6=Bluetooth.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceList = SagaSDK.TMSiGetDeviceList
    TMSiGetDeviceList.restype = TMSiDeviceRetVal
    TMSiGetDeviceList.argtype = [POINTER(TMSiDevList), c_int, c_uint, c_uint]

    #---
    # @details This command is used to open a device. This will create a connection
    # between API and DS and "lock" the interface between DR and DS.
    #
    # @Pre @li No device should have been openend.
    # @li TMSiGetDeviceList should have been called to obtain valid
    #       TMSiDeviceID
    #
    # @Post After TMSI_OK device is in "Device_Open".
    #
    # @depends Low level call 0x0102.
    #
    # @param[out] TMSiDeviceHandle  Handle to device use for further API calls.
    # @param[in] DeviceID Device to open, retrieved by "TMSiGetDeviceList".
    # @param[in] DRInterfaceType Interface to DR to use and lock. 0 = Unknown,1=USB, 2=Network, 3=WiFi, 4=Electrical, 5=Optical, 6=Bluetooth.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiOpenDevice = SagaSDK.TMSiOpenDevice
    TMSiOpenDevice.restype = TMSiDeviceRetVal
    TMSiOpenDevice.argtype = [POINTER(c_void_p), c_uint, c_uint]


    #---
    # @details This command is used to Close a device.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post After TMSI_OK the device STATEMACHINE is in "Device_Close" state.
    #
    # @depends Low level call 0x0103.
    #
    # @param[out] TMSiDeviceHandle  Handle of device to close.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiCloseDevice = SagaSDK.TMSiCloseDevice
    TMSiCloseDevice.restype = TMSiDeviceRetVal
    TMSiCloseDevice.argtype = [c_void_p]


    #---
    # @details This command is used to retrieve a status report from a TMSi device.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    # @Post No device change.
    #
    # @depends Low level call 0x0201.
    #
    # @param[in] TMSiDeviceHandle   Handle to the current open device.
    # @param[out] DeviceStatus      Status report of the connected device.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceStatus = SagaSDK.TMSiGetDeviceStatus
    TMSiGetDeviceStatus.restype = TMSiDeviceRetVal
    TMSiGetDeviceStatus.argtype = [c_void_p, POINTER(TMSiDevStatReport)]

    #---
    # @details This command is used to retrieve a full status report from a TMSi
    # device.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No device change.
    #
    # @depends Low level call 0x0202.
    #
    # @param[in] TMSiDeviceHandle   Handle to the current open device.
    # @param[out] FullDeviceStatus  Status report.
    # @param[out] DeviceBatteryReportList   list of BatteryReport(s).
    # @param[in] BatteryStatusListLen   Nr of BatteryReportLists allocated.
    #
    # @param[out] DeviceTime Device time report.
    # @param[out] StorageReport Device storage report.
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetFullDeviceStatus(void* TMSiDeviceHandle, TMSiDevFullStatReportType* FullDeviceStatus, TMSiDevBatReportType* DeviceBatteryStatusList, int32_t BatteryStatusListLen, TMSiTimeType* DeviceTime, TMSiDevStorageReportType* StorageReport);
    TMSiGetDeviceConfig = SagaSDK.TMSiGetFullDeviceStatus
    TMSiGetDeviceConfig.restype = TMSiDeviceRetVal
    TMSiGetDeviceConfig.argtype = [c_void_p, POINTER(TMSiDevFullStatReport), POINTER(TMSiDevBatReport), c_int, POINTER(TMSiTime), POINTER(TMSiDevStorageReport)]

    #---
    # @details This command is used to retrieve the current configuration from a
    # TMSi device. The response can be used to calculate the expected data streams
    # from the device. If a channel is enabled for sampling the "ChanDivider > -1".
    # If a channel will send its impedance data during an impedance measurement is
    # determined by the "ImpDivider".  The last channel in the device list (after
    # the SAW channel) is the PGND, this channel is included only for use during
    # impedance mode. Therefore the ChanDivider = -1 and the ImpDivider is > -1 for
    # this channel. To calculate the expected data stream in a certain mode multiply
    # all enabled channels with the sample frequency and int32_t.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No device change.
    #
    # @depends Low level call 0x0203.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] RecorderConfiguration Device configuration.
    # @param[out] ChannelsList  Channel(s) configuration.
    # @param[in] ChannelsListLen    Allocated nr of ChannelList items,
    # should be atleast NrOfChannels as mentioned in TMSiDevStatReportType
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceConfig = SagaSDK.TMSiGetDeviceConfig
    TMSiGetDeviceConfig.restype = TMSiDeviceRetVal
    TMSiGetDeviceConfig.argtype = [c_void_p, POINTER(TMSiDevGetConfig), POINTER(TMSiDevChDesc), c_int]

    #---
    # @details This command is used to set a new configuration on a TMSi
    # device.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post When a TMSI_OK is returned the Device configuration is updated. When an # error is returned the device configuration might not be updated. This should
    # be checked by requesting the current configuration \ref TMSiGetDeviceConfig.
    #
    # @depends Low level call 0x0204.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] RecorderConfiguration  Buffer with new configuration.
    # @param[in] ChannelConfigList      Buffer with ChannelConfigList items.
    # @param[in] ChannelConfigListLen   Nr of ChannelConfigList items.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDeviceConfig(void* TMSiDeviceHandle, TMSiDevSetConfigType* RecorderConfiguration, TMSiDevSetChCfgType* ChannelConfigList, int32_t ChannelConfigListLen);
    TMSiSetDeviceConfig = SagaSDK.TMSiSetDeviceConfig
    TMSiSetDeviceConfig.restype = TMSiDeviceRetVal
    TMSiSetDeviceConfig.argtype = [c_void_p, POINTER(TMSiDevSetConfig), POINTER(TMSiDevSetChCfg), c_int]

    #---
    # @details This command is used to set the time on a TMSi device.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post When a TMSI_OK is returned the internal time has been updated.
    #
    # @depends Low level call 0x0205.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] NewTime    Buffer with new time information.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDeviceRTC(void* TMSiDeviceHandle, TMSiTimeType* NewTime);
    TMSiSetDeviceRTC = SagaSDK.TMSiSetDeviceRTC
    TMSiSetDeviceRTC.restype = TMSiDeviceRetVal
    TMSiSetDeviceRTC.argtype = [c_void_p, POINTER(TMSiTime)]

    #---
    # @details This command is used to get sensor information from channels which
    # support this feature.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0206.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] SensorList    Channel(s) configuration.
    #
    # @param[in] SensorsListLen Nr of SensorList elements allocated, should be
    # atleast NrOfSensors from TMSiDevGetConfigType.
    # @param[out] RetSensorsListLen  Nr of Sensor lists returned.
    #                   .
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetDeviceSensor(void* TMSiDeviceHandle, TMSiDevGetSensType* SensorsList, uint32_t SensorsListLen, uint32_t* RetSensorsListLen);
    TMSiGetDeviceSensor = SagaSDK.TMSiGetDeviceSensor
    TMSiGetDeviceSensor.restype = TMSiDeviceRetVal
    TMSiGetDeviceSensor.argtype = [c_void_p, POINTER(TMSiDevGetSens), c_uint, POINTER(c_uint)]

    #---
    # @details This command is used to set sensor options for channels which
    # support this feature.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post When TMSI_OK returned the device sensor configuration is as requested,
    # else the previous configuration is still valid.
    #
    # @depends Low level call 0x0207.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] SensorList    List of sensor configuration(s).
    # @param[out] SensorsListLen    Nr of SensorList items.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDeviceSensor(void* TMSiDeviceHandle, TMSiSetDevSensType* SensorList, int32_t SensorsListLen);

    #---
    # @details This command is used to control the sampling mode on a TMSi
    # device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open" or "Device_Sampling".
    #
    # @Post When TMSI_OK is returned the device is in the "Device_Sampling" or
    # "Device_Open" state depending on the requested StartStop flag.
    # Sampling data should be retrieved by calling TMSiGetDeviceData.
    # STOPWiFiStream will result in a stop of datastream the device will go to
    # ambulant recording mode, the connection to TMSiDevice shall be closed by the
    # application.
    #
    # @depends Low level call 0x0301.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] DeviceSamplingMode New device sampling configuration.
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiSetDeviceSampling = SagaSDK.TMSiSetDeviceSampling
    TMSiSetDeviceSampling.restype = TMSiDeviceRetVal
    TMSiSetDeviceSampling.argtype = [c_void_p, POINTER(TMSiDevSampleReq)]

    #---
    # @details This command is used to set the device impedance mode.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open" or "Device_Impedance".
    #
    # @Post When TMSI_OK is returned the device is in the "Device_Impedance" or
    # "Device_Open" state depending on the requested StartStop flag.
    # Impedance data should be retrieved by calling TMSiGetDeviceData.
    #
    # @depends Low level call 0x0302.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] DeviceImpedanceMode    New device impedance configuration.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDeviceImpedance(void* TMSiDeviceHandle, TMSiDevImpReqType* DeviceImpedanceMode);
    TMSiSetDeviceImpedance = SagaSDK.TMSiSetDeviceImpedance
    TMSiSetDeviceImpedance.restype = TMSiDeviceRetVal
    TMSiSetDeviceImpedance.argtype = [c_void_p, POINTER(TMSiDevImpReq)]

    #---
    # @details This command is used to get the device streaming data. The
    # application can retrieve sampledata/impdata from the device. It returns data
    # as 32-bit float values, all data is already processed, meaning it is converted
    # from bits to units (as specified in the channel descriptor). The function will # return a buffer with a NrOfSets of samples, for each ENABLED channel one
    # sample per set. The application should match each sample with the
    # corresponding channel. All samples are in order of enabled channels starting
    # at the first channel.
    # The DataType indicates if the data is Sampledata DataType = 1,  ImpedanceData
    # DataType = 2, Sampledata Recording = 3.
    # In case of impedance data only Channels with "ImpDivider" > -1 are transmitted.
    # The buffer retured is a multiple of Samplesets.
    #
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle. The device shall be in "Device_Sampling" or
    # "Device_Impedance" state.
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0303.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] DeviceData       Received device Data.
    # @param[in] DeviceDataBufferSize      Buffersize for device Data;
    # @param[out] NrOfSets     The returned samplesets in this buffer
    # @param[out] DataType     The returned data type.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceData = SagaSDK.TMSiGetDeviceData
    TMSiGetDeviceData.restype = TMSiDeviceRetVal
    TMSiGetDeviceData.argtype = [c_void_p, POINTER(c_float), c_uint, POINTER(c_uint), POINTER(c_int)]

    #---
    # @details This command is used to get the current status of the streaming
    # databuffer. It returns the current value of the amount of data waiting in the
    # buffer.
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle. The device shall be in "Device_Sampling" or
    # "Device_Impedance" state.
    #
    # @Post No change in device state.
    #
    # @depends None, API call only.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] DeviceDataBuffered  The amount of data buffered for this device in
    # Bytes.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetDeviceDataBuffered(void* TMSiDeviceHandle, int32_t* DeviceDataBuffered);

    #---
    # @details This command is used to reset the internal data buffer thread for the
    # specified device after it has been stopped sampling.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends None, API call only.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DLL error received.
    #---
    TMSiResetDeviceDataBuffer = SagaSDK.TMSiResetDeviceDataBuffer
    TMSiResetDeviceDataBuffer.restype = TMSiDeviceRetVal
    TMSiResetDeviceDataBuffer.argtype = [c_void_p]

    #---
    # @details This command is used to get the device storage list.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0304.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] RecordingsList   List of available recordings on data recorder.
    # @param[in] RecordingsListLen     Buffersize for RecordingsList
    # @param[out] RetRecordingListLen   The amount of returned recordings in the list.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---

    TMSiGetDeviceStorageList = SagaSDK.TMSiGetDeviceStorageList
    TMSiGetDeviceStorageList.restype = TMSiDeviceRetVal
    TMSiGetDeviceStorageList.argtype = [c_void_p, POINTER(TMSiDevRecList), c_uint, POINTER(c_uint)]

    #---
    # @details This command is used to get a recorded file from the data recorder
    # device. The file is selected by using the "RecFileID" as returned by
    # "TMSiGetDeviceStorageList". After a successful return from this call the file
    # sample data can be retrieved by calling the "TMSiGetDeviceData" where the
    # "DataType" flag will be 3.  To Stop / abort the file transfer this call can be
    # used with the apropriate StartStop flag.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open" or "Device_Amb_Data".
    #
    # @Post If TMSI_OK the device STATEMACHINE shall be in "Device_Open" or
    # "Device_Amb_Data".
    #
    # @depends Low level call 0x0305.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] RecFileID     The file ID which should be retrieved.
    # @param[in] StartStop;  flag to start (1) and stop (0) the file transfer.
    # @param[out] RecordingMetaData Metadata of requested recording file.
    # @param[out] ImpedanceReportList      An impedance value for each impedance
    # enabled channel.
    # @param[in] ImpedanceReportListLen   Size of mpedanceReportListshould be
    # atleast the size of all channels with ImpDivider > -1 * int32_t. ImpDivider is
    # found in TMSiDevChDescType.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetRecordingFile(void* TMSiDeviceHandle, uint16_t RecFileID, uint16_t StartStop, TMSiDevRecDetailsType* RecordingMetaData, TMSiDevImpReportType* ImpedanceReportList, int32_t ImpedanceReportListLen);

    #---
    # @details This command is used to get a ambulant recording configuration from
    # the data recorder device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    # *
    # @Post No change in device state.
    #
    # @depends Low level call 0x0306.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] AmbulantConfiguration    The current ambulant configuration.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetDeviceAmbConfig(void* TMSiDeviceHandle, TMSiDevRecCfgType* AmbulantConfiguration);

    #---
    # @details This command is used to set a new ambulant recording configuration
    # from the data recorder device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0307.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] AmbulantConfiguration    The new ambulant configuration.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal  TMSiSetDeviceAmbConfig(void* TMSiDeviceHandle, TMSiDevRecCfgType* AmbulantConfiguration);

    #---
    # @details This command is used to get repair data from a device after a measurement.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open" or "Device_Repair".
    # @Post No change in device state.
    #
    # @depends Low level call 0x0308.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] RepairDataBuffer     Buffer to copy the requested repairdata into.
    # @param[in] RepairDataBufferSize   The available buffersize in bytes.
    # @param[out] NrOfSamples  The returned number of samples.
    # @param[in] RepairInfo The repair request
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal  TMSiGetDeviceRepairData(void* TMSiDeviceHandle, float* RepairDataBuffer, int32_t RepairDataBufferSize, int32_t* NrOfSamples, TMSiDevRepairReqType* RepairInfo);

    #---
    # @details This command is used to get calibration data from a device. All
    # available channels will be returned.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    # @Post No change in device state.
    #
    # @depends Low level call 0x0601.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] ChCalValuesList      The list of calibration data.
    # @param[in] ChCalValuesListLen    The amount of allocatedlist items.
    # @param[out] RetChCalValuesListLen    The returned list size.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetDeviceCalibration(void* TMSiDeviceHandle, TMSiDevChCalType* ChCalValuesList, int32_t ChCalValuesListLen, int32_t* RetChCalValuesListLen);

    #---
    # @details This command is used to set calibration data for a device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0602.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] ChCalValuesList       The list of new calibration data.
    # @param[in] ChCalValuesListLen    The size of ChCalValueList.
    #
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDeviceCalibration(void* TMSiDeviceHandle, TMSiDevChCalType* ChCalValuesList, int32_t ChCalValuesListLen);

    #---
    # @details This command is used to set calibration mode for a device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post Device is in "Device_Calibration".
    #
    # @depends Low level call 0x0603.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] SetCalibrationMode    The Calibration mode setting.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDeviceCalibrationMode(void* TMSiDeviceHandle, int32_t SetCalibrationMode);

    #---
    # @details This command is used to get the current logging activity and health state of the device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0604.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] DeviceDiagnostics The short health status report and current logging settings of the device.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetDeviceDiagnostics(void* TMSiDeviceHandle, TMSiDevGetDiagStatType* DeviceDiagnostics);

    #---
    # @details This command is used to set the logging options for a device. When
    # logging is enabled without a logfile reset, the device will append to an
    # existing log file.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0605.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] DeviceDiagnosticsCfg The short health status report of the device.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDeviceDiagnostics(void* TMSiDeviceHandle, TMSiDevSetDiagStatType* DeviceDiagnosticsCfg);

    #---
    # @details This command is used to get the current logfile of a device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0606.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] TMSiDevice The device of which the log file is requested.
    # @param[in] DeviceLogBufferSize The size of the DeviceLogData buffer;
    # @param[out] RetDeviceLogBufferSize The size of the returned log
    # @param[out] DeviceLogData The logdata from the device.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetDeviceLog(void* TMSiDeviceHandle, uint32_t TMSiDevice, uint32_t DeviceLogBufferSize, uint32_t* RetDeviceLogBufferSize, uint8_t* DeviceLogData);

    #---
    # @details This command is used to get the current firmware status of a device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0607.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] TMSiDevice The device of which the firmware status is requested.
    # @param[out] FWReport The current firmware status report.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetDeviceFWStatus(void* TMSiDeviceHandle, uint32_t TMSiDevice, TMSiDevFWStatusReportType* FWReport);

    #---
    # @details This command is used to prepare a device for a firmware update. This
    # call sends the firmware header which is checked for compatibility.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0608.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] TMSiDevice The device which needs to prepare for a firmware update.
    # @param[in] NewFWHeader The new firmware header for the firmware.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful, firmware header is accepted.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDeviceFWUpdate(void* TMSiDeviceHandle, uint32_t TMSiDevice, TMSiFWHeaderFileType* NewFWHeader);

    #---
    # @details This command is used to send the firmware data to a device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0609.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] TMSiDevice The destination device for the firmware update.
    # @param[in] FWDataSize The size of the firmware data in bytes.
    # @param[in] FWData The new firmware data.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiPushFWUpdate(void* TMSiDeviceHandle, uint32_t TMSiDevice, uint32_t FWDataSize, uint8_t* FWData);

    #---
    # @details This command is used to initiate or abort a firmware update.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x060A.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] TMSiDevice The device which is being updated.
    # @param[in] FWAction The action to perform on the device.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiDoneFWUpdate(void* TMSiDeviceHandle, uint32_t TMSiDevice, int32_t FWAction);

    #---
    # @details This command is used to program production information during
    # manufacturing.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x060B.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] ProductConfig The device which is being configured.
    # @param[in] ChannelConfigList The list of channels to configure.
    # @param[in] ChannelConfigListLen The size of the ChannelConfigList.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetProductConfig(void* TMSiDeviceHandle, TMSiDevProductConfigType* ProductConfig, TMSiDevProductChCfgType* ChannelConfigList, uint32_t ChannelConfigListLen);

    #---
    # @details This command is used to set the network configuration for the DS.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x060C.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] GetDSNetworkConig The current network configuration of the DS.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetDevNetworkConfig(void* TMSiDeviceHandle, TMSiDevNetworkConfigType* GetDSNetworkConig);

    #---
    # @details This command is used to set the network configuration for the DS.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x060D.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] SetDSNetworkConig The new network configuration for the DS.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDevNetworkConfig(void* TMSiDeviceHandle, TMSiDevNetworkConfigType* SetDSNetworkConig);



    #endif
