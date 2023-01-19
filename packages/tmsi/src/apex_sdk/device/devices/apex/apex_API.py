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
 * @file apex_API.py 
 * @brief 
 * API calls to communication library.
 */


'''

from ctypes import *
from sys import platform
import os
from array import *

from .apex_API_enums import *
from .apex_API_structures import *
from ....tmsi_utilities.tmsi_logger import TMSiLogger

DeviceHandle = c_void_p
TMSiDeviceHandle = DeviceHandle(0)
ApexDllAvailable = False
ApexDllLocked = True

if platform == "win32": # Windows
    search_path = "C:/Program files/TMSi/APEX"
    name = "TMSiApexDeviceLib.dll"
    result = os.path.join(search_path, name)
    so_name = os.path.abspath(result)
    if os.path.exists(so_name):
        TMSiLogger().debug("{} available.".format(so_name))
        ApexDllAvailable = True
    try:
        ApexSDK = CDLL(so_name)
        ApexDllLocked = False
        sdk_handle = ApexSDK._handle
        TMSiLogger().debug("Successfully loaded Apex device library, handle: " + hex(sdk_handle) )
    except OSError as e:
        if ApexDllAvailable:
            TMSiLogger().warning("{} already in use.".format(so_name))
else:
    TMSiLogger().warning("Unsupported platform.")

if ApexDllAvailable and not ApexDllLocked:
    # DLL interface

    #---
    # @details This command is used to retrieve a list of available TMSi devices
    # connected to the PC. This query is performed on the "DRInterfaceType" specified
    # by the user. All other interface types are ignored.
    #
    # @Pre \ref No device should have been opened. Device is in Close state.
    #
    # @Post No device change.
    #
    # @param[out] TMSiDeviceList    List of found devices.
    # @param[out] NrOfFoundDevices  Number of found devices.
    # @param[in]  DRInterfaceType   See <TMSiInterface>
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceList = ApexSDK.TMSiGetDeviceList
    TMSiGetDeviceList.restype = TMSiDeviceRetVal
    TMSiGetDeviceList.argtype = [POINTER(TMSiDevList), c_uint, c_uint, POINTER(c_uint)]

    #---
    # @details This command is used to open a interface. This will create a connection
    # between API and DR and "lock" the interface.
    #
    # @Pre @li No interface should have been openend.
    # @li TMSiGetDeviceList should have been called to obtain valid
    #       TMSiDeviceID
    #
    # @Post After TMSI_OK device is in "Device_Open".
    #
    # @param[out] TMSiDeviceHandle  Handle to device use for further API calls.
    # @param[in]  DongleID          Dongle to use in case of the bluetooth interface, retrieved by "TMSiGetDongleList".
    # @param[in]  DeviceID          Device to open, retrieved by "TMSiGetDeviceList".
    # @param[in]  DRInterfaceType   See <TMSiInterface>
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiOpenInterface = ApexSDK.TMSiOpenInterface
    TMSiOpenInterface.restype = TMSiDeviceRetVal
    TMSiOpenInterface.argtype = [POINTER(c_void_p), c_ushort, c_ushort, c_uint]


    #---
    # @details This command is used to close an interface.
    #
    # @Pre \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post After TMSI_OK the device STATEMACHINE is in "Device_Close" state.
    #
    # @param[in] TMSiDeviceHandle  Handle of device to close.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiCloseInterface = ApexSDK.TMSiCloseInterface
    TMSiCloseInterface.restype = TMSiDeviceRetVal
    TMSiCloseInterface.argtype = [c_void_p]


    #---
    # @details This command is used to retrieve a list of bluetooth-dongles
    # connected to the PC.
    #
    # @Post No device change.
    #
    # @param[out] DongleList        List of found bluetooth dongles.
    # @param[in]  DongleListLen     Max num of items in <DongleList>
    # @param[out] RetDongleListLen  Number of found dongles.
    #
    # @return
    # @li TMSiStatusOK Ok, if dongles have been found.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDongleList = ApexSDK.TMSiGetDongleList
    TMSiGetDongleList.restype = TMSiDeviceRetVal
    TMSiGetDongleList.argtype = [POINTER(TMSiDongleList), c_uint, POINTER(c_uint)]


    #---
    # @details This command is used to pair a PC-BT-dongle with a Apex recorder.
    #
    # @param[in] DongleID, retrieved by "TMSiGetDongleList".
    # @param[in] DeviceID, retrieved by "TMSiGetDeviceList".
    #
    # @return
    # @li TMSiStatusOK Ok, if pairing successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiPairDevice = ApexSDK.TMSiPairDevice
    TMSiPairDevice.restype = TMSiDeviceRetVal
    TMSiPairDevice.argtype = [c_ushort, c_ushort]


    #---
    # @details This command is used to retrieve an info report from a TMSi device.
    #
    # @Pre \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    # @Post No device change.
    #
    # @param[in] TMSiDeviceHandle   Handle to the current open device.
    # @param[out] DeviceInfo      Status report of the connected device.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceInfo = ApexSDK.TMSiGetDeviceInfo
    TMSiGetDeviceInfo.restype = TMSiDeviceRetVal
    TMSiGetDeviceInfo.argtype = [c_void_p, POINTER(TMSiDevInfoReport)]

    #---
    # @details This command is used to retrieve an power status report from a TMSi
    # device.
    #
    # @Pre \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    # @Post No device change.
    #
    # @param[in] TMSiDeviceHandle   Handle to the current open interface.
    # @param[out] DevicePowerStatus Power Status report of the connected device.
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDevicePowerStatus = ApexSDK.TMSiGetDevicePowerStatus
    TMSiGetDevicePowerStatus.restype = TMSiDeviceRetVal
    TMSiGetDevicePowerStatus.argtype = [c_void_p, POINTER(TMSiDevInfoReport)]


    #---
    # @details This command is used to get the device interface status.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    # @Post No change in device state.
    #
    # @param[in]  TMSiDeviceHandle  Handle to the current open device.
    # @param[in]  InterfaceType      See <TMSiInterfaceType>.
    # @param[out] InterfaceStatus   Holds the returned device interface status.
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceInterfaceStatus = ApexSDK.TMSiGetDeviceInterfaceStatus
    TMSiGetDeviceInterfaceStatus.restype = TMSiDeviceRetVal
    TMSiGetDeviceInterfaceStatus.argtype = [c_void_p, c_uint, POINTER(TMSiInterfaceStatus)]


    #---
    # @details This command is used to set the device interface configuration.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    # @Post No change in device state.
    #
    # @param[in] TMSiDeviceHandle   Handle to the current open device.
    # @param[in] InterfaceType      See <TMSiInterfaceType>.
    # @param[in] InterfaceControl   See <TMSiControlType>.
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiSetDeviceInterfaceConfig = ApexSDK.TMSiSetDeviceInterfaceConfig
    TMSiSetDeviceInterfaceConfig.restype = TMSiDeviceRetVal
    TMSiSetDeviceInterfaceConfig.argtype = [c_void_p, c_uint, c_byte]


    #---
    # @details This command is used to retrieve the actual sampling configuration from a TMSi
    # device.
    #
    # @Pre \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No device change.
    #
    # @param[in] TMSiDeviceHandle   Handle to the current open device.
    # @param[out] DevSamplingCfg  Actual sampling configuration.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceSamplingConfig = ApexSDK.TMSiGetDeviceSamplingConfig
    TMSiGetDeviceSamplingConfig.restype = TMSiDeviceRetVal
    TMSiGetDeviceSamplingConfig.argtype = [c_void_p, POINTER(TMSiDevSamplingCfg)]

    #---
    # @details This command is used to set the actual sampling configuration from a TMSi
    # device.
    #
    # @Pre \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No device change.
    #
    # @param[in] TMSiDeviceHandle   Handle to the current open device.
    # @param[out] DevSamplingCfg  Sampling configuration to set.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiSetDeviceSamplingConfig = ApexSDK.TMSiSetDeviceSamplingConfig
    TMSiSetDeviceSamplingConfig.restype = TMSiDeviceRetVal
    TMSiSetDeviceSamplingConfig.argtype = [c_void_p, POINTER(TMSiDevSamplingCfg)]

    #---
    # @details This command is used to get the channel configuration data from a device. All
    # available channels will be returned.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    # @Post No change in device state.
    #
    # @param[in]  TMSiDeviceHandle  Handle to the current open device.
    # @param[out] DevChNameList     The list of fixed channel names.
    # @param[out] DevAltChNameList  The list of alternate channel names.
    # @param[in]  ChannelListLen    The amount of allocated channel list items for both lists.
    # @param[out] RetChannelListLen Holds the number channels in the returned lists.
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceChannelConfig = ApexSDK.TMSiGetDeviceChannelConfig
    TMSiGetDeviceChannelConfig.restype = TMSiDeviceRetVal
    TMSiGetDeviceChannelConfig.argtype = [c_void_p, POINTER(TMSiDevChName), POINTER(TMSiDevAltChName), c_uint, POINTER(c_uint)]


    #---
    # @details This command is used to update alternate channel names.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    # @Post No change in device state.
    #
    # @param[in] TMSiDeviceHandle   Handle to the current open device.
    # @param[in] DevAltChNameList   The list of alternate channel names.
    # @param[in] ChannelListLen     Max num of items in <DevAltChNameList>.
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiSetDeviceChannelConfig = ApexSDK.TMSiSetDeviceChannelConfig
    TMSiSetDeviceChannelConfig.restype = TMSiDeviceRetVal
    TMSiSetDeviceChannelConfig.argtype = [c_void_p, POINTER(TMSiDevAltChName), c_uint]

    #---
    # @details This command is used to get reference status of the channels.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    # @Post No change in device state.
    #
    # @param[in]  TMSiDeviceHandle        Handle to the current open device.
    # @param[out] DevChanRefList          The channel reference list.
    # @param[in]  DevChanRefListLength    Max num of items in <DevChanRefList>.
    # @param[out] RetDevChanRefListLength Holds the number channels in the returned list.
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceReferenceConfig = ApexSDK.TMSiGetDeviceReferenceConfig
    TMSiGetDeviceReferenceConfig.restype = TMSiDeviceRetVal
    TMSiGetDeviceReferenceConfig.argtype = [c_void_p, POINTER(TMSiDevChanRef), c_uint, POINTER(c_uint)]


    #---
    # @details This command is used to set reference status of the channels.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    # @Post No change in device state.
    #
    # @param[in]  TMSiDeviceHandle      Handle to the current open device.
    # @param[out] DevChanRefList        The channel list with the references to set.
    # @param[in]  DevChanRefListLength  Max num of items in <DevChanRefList>.
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiSetDeviceReferenceConfig = ApexSDK.TMSiSetDeviceReferenceConfig
    TMSiSetDeviceReferenceConfig.restype = TMSiDeviceRetVal
    TMSiSetDeviceReferenceConfig.argtype = [c_void_p, POINTER(TMSiDevChanRef), c_uint]


    #---
    # @details This command is used to get the time off a TMSi device.
    #
    # @Pre \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post When a TMSI_OK is returned the internal time has been updated.
    #
    # @depends Low level call 0x0205.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] Time    Buffer for storing time information.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceRTC = ApexSDK.TMSiGetDeviceRTC
    TMSiGetDeviceRTC.restype = TMSiDeviceRetVal
    TMSiGetDeviceRTC.argtype = [c_void_p, POINTER(TMSiTime)]


    #---
    # @details This command is used to set the time on a TMSi device.
    #
    # @Pre \ref TMSiOpenInterface should have been called and returned a valid
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
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiSetDeviceRTC = ApexSDK.TMSiSetDeviceRTC
    TMSiSetDeviceRTC.restype = TMSiDeviceRetVal
    TMSiSetDeviceRTC.argtype = [c_void_p, POINTER(TMSiTime)]


    #---
    # @details This command is used to get the card recording configuration from
    # the data recorder.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @param[in]  TMSiDeviceHandle Handle to the current open device.
    # @param[out] DevCardRecCfg    The current card recording configuration.
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceCardRecordingConfig = ApexSDK.TMSiGetDeviceCardRecordingConfig
    TMSiGetDeviceCardRecordingConfig.restype = TMSiDeviceRetVal
    TMSiGetDeviceCardRecordingConfig.argtype = [c_void_p, POINTER(TMSiDevCardRecCfg)]


    #---
    # @details This command is used to set a new card recording configuration
    # to the data recorder.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @param[in]  TMSiDeviceHandle  Handle to the current open device.
    # @param[out] DevCardRecCfg    The new card recording configuration.
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiSetDeviceCardRecordingConfig = ApexSDK.TMSiSetDeviceCardRecordingConfig
    TMSiSetDeviceCardRecordingConfig.restype = TMSiDeviceRetVal
    TMSiSetDeviceCardRecordingConfig.argtype = [c_void_p, POINTER(TMSiDevCardRecCfg)]


    #---
    # @details This command is used to clear the contents of the memory card of the device.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    #
    # @return
    # @li TMSiStatusOK Ok, if factory defaults were set successful.
    # @li Any TMSI_DR*, TMSI_DLL error otherwise.
    #---
    TMSiResetDeviceCard = ApexSDK.TMSiResetDeviceCard
    TMSiResetDeviceCard.restype = TMSiDeviceRetVal
    TMSiResetDeviceCard.argtype = [c_void_p]


    #---
    # @details This command is used to set the device back to its factory defaults.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    #
    # @return
    # @li TMSiStatusOK Ok, if factory defaults were set successful.
    # @li Any TMSI_DR*, TMSI_DLL error otherwise.
    #---
    TMSiSetDeviceFactoryDefaults = ApexSDK.TMSiSetDeviceFactoryDefaults
    TMSiSetDeviceFactoryDefaults.restype = TMSiDeviceRetVal
    TMSiSetDeviceFactoryDefaults.argtype = [c_void_p]



    #---
    # @details This command is used to get the sample metadata-header from a device.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    # @Post No change in device state.
    #
    # @param[in]  TMSiDeviceHandle      Handle to the current open device.
    # @param[out] DRInterfaceType       Interface type (USB or bluetooth) to retrieve the sample
    #                                   metadata header from.
    # @param[out] SampleMetadataHeader  Sample metadata header.
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceSampleMetadataHeader = ApexSDK.TMSiGetDeviceSampleMetadataHeader
    TMSiGetDeviceSampleMetadataHeader.restype = TMSiDeviceRetVal
    TMSiGetDeviceSampleMetadataHeader.argtype = [c_void_p, c_uint, POINTER(TMSiSampleMetadataHeader)]


    #---
    # @details This command is used to get the channel meta data from a device.
    # The metadata of available channels will be returned.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    # @Post No change in device state.
    #
    # @param[in]  TMSiDeviceHandle                  Handle to the current open device.
    # @param[out] DRInterfaceType                   Interface type (USB or bluetooth) to retrieve the channel
    #                                               meta data from.
    # @param[out] ChannelMetadataList               The allocated channel metadata list.
    # @param[in]  ChannelListLen                    The amount of allocated channel list items.
    # @param[out] RetChannelListLen                 The returned nr of channel list items.
    # @param[out] CyclingStateMetadataList          The allocated Cycling-State metadata list.
    # @param[in]  CyclingStateMetadataListLen       The amount of allocated Cycling-State list items.
    # @param[out] RetCyclingStatMetadataListLen     The returned nr of Cycling-State list items.
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceSampleMetadata = ApexSDK.TMSiGetDeviceSampleMetadata
    TMSiGetDeviceSampleMetadata.restype = TMSiDeviceRetVal
    TMSiGetDeviceSampleMetadata.argtype = [c_void_p, c_uint, POINTER(TMSiChannelMetadata), c_uint, POINTER(c_uint), POINTER(TMSiCyclingStateMetadata), c_uint, POINTER(c_uint)]


    #---
    # @details This command is used to get the impedance meta data from a device.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    # @Post No change in device state.
    #
    # @param[in]  TMSiDeviceHandle      Handle to the current open device.
    # @param[out] ImpedanceMetadata     The allocated impedance metadata structs
    # @param[in]  ImpedanceListLen      The amount of allocated impedance-channel list items.
    # @param[out] RetImpedanceListLen   The returned nr of impdance-channel list items.
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceImpedanceMetadata = ApexSDK.TMSiGetDeviceImpedanceMetadata
    TMSiGetDeviceImpedanceMetadata.restype = TMSiDeviceRetVal
    TMSiGetDeviceImpedanceMetadata.argtype = [c_void_p, POINTER(TMSiImpedanceMetadata), c_uint, POINTER(c_uint)]



    #---
    # @details This command is used to control the sampling mode on a TMSi
    # device.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open" or "Device_Sampling".
    #
    # @Post When TMSiStatusOK is returned the device is in the "Device_Sampling" or
    # "Device_Open" state depending on the requested StartStop flag.
    # Sampling data should be retrieved by calling TMSiGetDeviceData.
    #
    # @param[in] TMSiDeviceHandle   Handle to the current open device.
    # @param[in] DevSampleRequest   New device sampling request.
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiSetDeviceSamplingRequest = ApexSDK.TMSiSetDeviceSamplingRequest
    TMSiSetDeviceSamplingRequest.restype = TMSiDeviceRetVal
    TMSiSetDeviceSamplingRequest.argtype = [c_void_p, POINTER(TMSiDevSampleRequest)]

    #---
    # @details This command is used to control the impedance mode on a TMSi
    # device.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open" or "Device_Impedance".
    #
    # @Post When TMSiStatusOK is returned the device is in the "Device_Impedance" or
    # "Device_Open" state depending on the requested StartStop flag.
    # Impedance data should be retrieved by calling TMSiGetDeviceData.
    #
    # @param[in] TMSiDeviceHandle       Handle to the current open device.
    # @param[in] DevImpedanceRequest    New device impedance request.
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiSetDeviceImpedanceRequest = ApexSDK.TMSiSetDeviceImpedanceRequest
    TMSiSetDeviceImpedanceRequest.restype = TMSiDeviceRetVal
    TMSiSetDeviceImpedanceRequest.argtype = [c_void_p, POINTER(TMSiDevImpedanceRequest)]

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
    TMSiGetDeviceData = ApexSDK.TMSiGetDeviceData
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
    TMSiResetDeviceDataBuffer = ApexSDK.TMSiResetDeviceDataBuffer
    TMSiResetDeviceDataBuffer.restype = TMSiDeviceRetVal
    TMSiResetDeviceDataBuffer.argtype = [c_void_p]


    #---
    # @details This command is used to get the device streaming impedance data. The
    # application can retrieve impedance sample data from the device.
    # The function will return a buffer with a NrOfSets of impedance samples, for each channel one
    # impedance-sample per set. The application should match each sample with the
    # corresponding channel. All samples are in order of enabled channels starting
    # at the first channel.
    # The buffer retured is a multiple of Samplesets.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle. The device shall be in "Device_Impedance" state.
    #
    # @Post No change in device state.
    #
    # @param[in]  TMSiDeviceHandle      Handle to the current open device.
    # @param[out] ImpedanceSampleBuffer Received impedance-sample data.
    # @param[in]  DeviceDataBufferSize  Buffersize for impedance-sample data;
    # @param[out] NrOfSets              The returned samplesets in this buffer
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---

    TMSiGetDeviceImpedanceData = ApexSDK.TMSiGetDeviceImpedanceData
    TMSiGetDeviceImpedanceData.restype = TMSiDeviceRetVal
    TMSiGetDeviceImpedanceData.argtype = [c_void_p, POINTER(TMSiImpedanceSample), c_uint, POINTER(c_uint)]

    #---
    # @details This command is used to get the card status.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @param[in]  TMSiDeviceHandle          Handle to the current open device.
    # @param[out] CardStatus                Status of the card on the data recorder.
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceCardStatus = ApexSDK.TMSiGetDeviceCardStatus
    TMSiGetDeviceCardStatus.restype = TMSiDeviceRetVal
    TMSiGetDeviceCardStatus.argtype = [c_void_p, POINTER(TMSiDevCardStatus)]


    #---
    # @details This command is used to get the card recording list.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @param[in]  TMSiDeviceHandle          Handle to the current open device.
    # @param[out] CardFileList              List of available card recordings on data recorder.
    # @param[in]  CardFileListListLen       Buffersize for CardRecordingsList
    # @param[out] RetCardFileListListLen    The amount of returned card recordings in the list.
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceCardFileList = ApexSDK.TMSiGetDeviceCardFileList
    TMSiGetDeviceCardFileList.restype = TMSiDeviceRetVal
    TMSiGetDeviceCardFileList.argtype = [c_void_p, POINTER(TMSiDevCardFileInfo), c_uint, POINTER(c_uint)]


    #---
    # @details This command is used to get the file metadata-header from a file.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    # @Post No change in device state.
    #
    # @param[in]  TMSiDeviceHandle      Handle to the current open device.
    # @param[out] FileMetadataHeader    File metadata header.
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceCardFileMetadataHeader = ApexSDK.TMSiGetDeviceCardFileMetadataHeader
    TMSiGetDeviceCardFileMetadataHeader.restype = TMSiDeviceRetVal
    TMSiGetDeviceCardFileMetadataHeader.argtype = [c_void_p, c_ushort, POINTER(TMSiFileMetadataHeader)]


    #---
    # @details This command is used to get the sample meta data from a specific card recording.
    # The metadata of available channels within the card recording will be returned.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    # @Post No change in device state.
    #
    # @param[in]  TMSiDeviceHandle                  Handle to the current open device.
    # @param[in]  RecFileID                         Unique ID of the card recording retrieved with TMSiGetDeviceCardRecordingList().
    # @param[out] DevCardFileDetails                Holds the returned card recording details.
    #                                               meta data from.
    # @param[out] ChannelMetadataList               The allocated channel metadata list.
    # @param[in]  ChannelMetadataListLen            The amount of allocated channel list items.
    # @param[out] RetChannelMetadataListLen         Holds the number channels in the returned list.
    # @param[out] CyclingStateMetadataList          The allocated Cycling-State metadata list.
    # @param[in]  CyclingStateMetadataListLen       The amount of allocated Cycling-State list items.
    # @param[out] RetCyclingStatMetadataListLen     The returned nr of Cycling-State list items.
    # @param[out] ImpedanceReportList               The allocated impedance report list.
    # @param[in]  ImpedanceReportListLen            The amount of allocated impedance report list items.
    # @param[out] RetImpedanceReportListLen         Holds the number of impedance report in the returned list.
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceCardFileMetadata = ApexSDK.TMSiGetDeviceCardFileMetadata
    TMSiGetDeviceCardFileMetadata.restype = TMSiDeviceRetVal
    TMSiGetDeviceCardFileMetadata.argtype = [c_void_p, c_ushort, POINTER(TMSiDevCardFileDetails), POINTER(TMSiChannelMetadata), c_uint, POINTER(c_uint), POINTER(TMSiCyclingStateMetadata), c_uint, POINTER(c_uint), POINTER(TMSiDevImpReportMetadata), POINTER(TMSiDevImpReport), c_uint, POINTER(c_uint)]

    #---
    # @details This command is used to control the download of a card recording from a TMSi
    # device.
    #
    # @Pre @li \ref TMSiOpenInterface should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open" or "Device_Card_Recording_Data.
    #
    # @Post When TMSiStatusOK is returned the device is in the "Device_Card_Recording_Data" or
    # "Device_Open" state depending on the requested StartStop flag.
    # Sampling data should be retrieved by calling TMSiGetDeviceData.
    #
    # @param[in] TMSiDeviceHandle   Handle to the current open device.
    # @param[in] DevSetCardFileReq  New card recording download request.
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiSetDeviceCardFileRequest = ApexSDK.TMSiSetDeviceCardFileRequest
    TMSiSetDeviceCardFileRequest.restype = TMSiDeviceRetVal
    TMSiSetDeviceCardFileRequest.argtype = [c_void_p, POINTER(TMSiDevSetCardFileReq)]


    #---
    # @details This command is used to get the first event fom the event buffer thread.
    #
    # @Pre @li \ref None.
    #
    # @Post No change in device state.
    #
    # @depends None, API call only.
    #
    # @param[out] Event     Event details when event was present.
    #
    # @return
    # @li TMSiStatusOK Ok, if event is present.
    # @li TMSiStatusDllBufferError if no event present.
    #---
    TMSiGetEvent = ApexSDK.TMSiGetEvent
    TMSiGetEvent.restype = TMSiDeviceRetVal
    TMSiGetEvent.argtype = [POINTER(TMSiEvent)]


    #---
    # @details This command is used to get the number of events present in the internal event buffer thread.
    #
    # @Pre @li \ref None.
    #
    # @Post No change in device state.
    #
    # @depends None, API call only.
    #
    # @param[out] NumOfEvents   Number of events present in the internal event buffer.
    #
    # @return
    # @li TMSiStatusOK.
    #---
    TMSiGetEventBuffered = ApexSDK.TMSiGetEventBuffered
    TMSiGetEventBuffered.restype = TMSiDeviceRetVal
    TMSiGetEventBuffered.argtype = [POINTER(c_ushort)]

    #---
    # @details This command is used to reset the internal event buffer thread.
    #
    # @Pre @li \ref None.
    #
    # @Post No change in device state.
    #
    # @depends None, API call only.
    #
    # @param[in] None
    #
    # @return
    # @li TMSiStatusOK.
    #---
    TMSiResetEventBuffer = ApexSDK.TMSiResetEventBuffer
    TMSiResetEventBuffer.restype = TMSiDeviceRetVal
    TMSiResetEventBuffer.argtype = []

    #---
    # @details This command is used to retrieve the device driver version info report.
    #
    # @param[out] DevDrvInfoReport  device driver version info report.
    #
    # @return
    # @li TMSiStatusOK Ok, if response received successful.
    # @li Any TMSI_DR*, TMSI_DLL error received.
    #---

    TMSiGetDeviceDriverVersionInfo = ApexSDK.TMSiGetDeviceDriverVersionInfo
    TMSiGetDeviceDriverVersionInfo.restype = TMSiDeviceRetVal
    TMSiGetDeviceDriverVersionInfo.argtype = [POINTER(TMSiDeviceDriverVersionInfo)]