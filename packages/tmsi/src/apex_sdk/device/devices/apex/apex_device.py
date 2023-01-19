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
 * @file apex_device.py 
 * @brief 
 * APEX Device object.
 */


"""

import datetime
import os
import time

from ....tmsi_errors.error import TMSiError, TMSiErrorCode
from ....tmsi_utilities.decorators import LogPerformances

from ...tmsi_device import TMSiDevice
from ...tmsi_device_enums import *

from .apex_structures.apex_const import ApexConst
from .apex_structures.apex_info import ApexInfo
from .apex_structures.apex_config import ApexConfig
from .apex_structures.dongle_info import DongleInfo
from .apex_structures.apex_channel import ApexChannel
from .apex_structures.apex_impedance_channel import ApexImpedanceChannel
from .apex_API_structures import *
from .apex_API_enums import *
from .apex_dongle import ApexDongle

if "TMSi_ENV" in os.environ and os.environ["TMSi_ENV"] == "MOCKED":
    from .apex_API_mock import *
elif "TMSi_ENV" in os.environ and os.environ["TMSi_ENV"] == "MOCKED24":
    from .apex_API_mock_24 import *
else:
    from .apex_API import *


class ApexDevice(TMSiDevice):
    """A class to represent and handle the Apex"""

    __apex_sdk = None
    __MAX_NUM_DEVICES = 10
    __MAX_NUM_DONGLES = 2
    __device_info_list = []
    __dongle_info_list = []
    __DEVICE_TYPE = "APEX"

    @LogPerformances
    def __init__(
        self,
        serial_number,
        dr_interface: DeviceInterfaceType,
        pairing_status=PairingStatus.no_pairing_needed,
        dongle_serial_number=0,
        idx=-1,
    ):
        """Initialize the Device.

        :param dr_serial_number: Serial number of the device.
        :type dr_serial_number: int
        :param dr_interface: Interface to use to communicate with the device.
        :type dr_interface: DeviceInterfaceType
        :param dongle_serial_number: Serial number of the dongle, default to 0.
        :type dongle_serial_number: int
        :param idx: Device index, defaults to -1.
        :type idx: int, optional
        """
        self.__device_handle = DeviceHandle(0)
        self.__info = ApexInfo(
            id=idx,
            serial_number=serial_number,
            dr_interface=dr_interface,
            dongle_serial_number=dongle_serial_number,
            pairing_status=pairing_status,
        )
        self.__config = ApexConfig()

    @LogPerformances
    def close(self):
        """Closes the connection to the device.

        :raises TMSiError: TMSiErrorCode.device_error if impossible to close.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        if self.__info.get_state() != DeviceState.disconnected:
            self.__last_error_code = TMSiCloseInterface(self.__device_handle)
            self.__info.set_state(DeviceState.disconnected)
            if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
                return
            else:
                raise TMSiError(
                    TMSiErrorCode.device_error, self.__last_error_code
                )
        else:
            raise TMSiError(TMSiErrorCode.device_not_connected)

    @LogPerformances
    def discover(dr_interface: DeviceInterfaceType, num_retries: int = 3):
        """Discovers available devices.

        :param dr_interface: Device interface to be searched.
        :type dr_interface: DeviceInterfaceType
        :param num_retries: Number of retries, optional
        :type num_reties: int
        """
        if not ApexDllAvailable:
            TMSiLogger().warning("APEX DLL not available.")
            raise TMSiError(error_code=TMSiErrorCode.missing_dll)
        if ApexDllLocked:
            TMSiLogger().warning("APEX DLL already in use.")
            raise TMSiError(error_code=TMSiErrorCode.already_in_use_dll)
        ApexDevice.get_sdk()  # if sdk already available, nothing happens, otherwise initialize
        for i in range(ApexDevice.__MAX_NUM_DEVICES):
            if (
                ApexDevice.__device_info_list[i].get_dr_interface()
                == dr_interface
            ):
                ApexDevice.__device_info_list[i] = ApexInfo()

        for i in range(ApexDevice.__MAX_NUM_DONGLES):
            ApexDevice.__dongle_info_list[i] = DongleInfo()

        device_list = (TMSiDevList * ApexDevice.__MAX_NUM_DEVICES)()
        dongle_list = (TMSiDongleList * ApexDevice.__MAX_NUM_DONGLES)()
        num_found_devices = (c_uint)(0)
        num_found_dongles = (c_uint)(0)

        ret = TMSiGetDongleList(
            pointer(dongle_list),
            ApexDevice.__MAX_NUM_DONGLES,
            pointer(num_found_dongles),
        )
        if ret == TMSiDeviceRetVal.TMSiStatusOK:
            for i in range(ApexDevice.__MAX_NUM_DONGLES):
                if i < num_found_dongles.value:
                    ApexDevice.__dongle_info_list[i].TMSiDongleID = dongle_list[
                        i
                    ].TMSiDongleID
                    ApexDevice.__dongle_info_list[i].SerialNumber = dongle_list[
                        i
                    ].SerialNumber
                else:
                    break

        for i in range(ApexDevice.__MAX_NUM_DEVICES):
            device_list[i].TMSiDeviceID = ApexConst.TMSI_DEVICE_ID_NONE

        while num_retries > 0:
            ret = TMSiGetDeviceList(
                pointer(device_list),
                ApexDevice.__MAX_NUM_DEVICES,
                dr_interface.value,
                pointer(num_found_devices),
            )

            if ret == TMSiDeviceRetVal.TMSiStatusOK:
                # Devices are found, update the local device list with the found result
                for i in range(ApexDevice.__MAX_NUM_DEVICES):
                    if (
                        device_list[i].TMSiDeviceID
                        != ApexConst.TMSI_DEVICE_ID_NONE
                    ):
                        for ii in range(ApexDevice.__MAX_NUM_DEVICES):
                            if (
                                ApexDevice.__device_info_list[ii].get_id()
                                == ApexConst.TMSI_DEVICE_ID_NONE
                            ):
                                ApexDevice.__device_info_list[ii].set_id(
                                    device_list[i].TMSiDeviceID
                                )
                                ApexDevice.__device_info_list[
                                    ii
                                ].set_dr_interface(dr_interface)
                                ApexDevice.__device_info_list[
                                    ii
                                ].set_dr_serial_number(
                                    device_list[i].SerialNumberDataRecorder
                                )
                                ApexDevice.__device_info_list[
                                    ii
                                ].set_dongle_serial_number(
                                    device_list[i].SerialNumberDongle
                                )
                                ApexDevice.__device_info_list[
                                    ii
                                ].set_pairing_status(
                                    device_list[i].PairingStatus
                                )
                                ApexDevice.__device_info_list[ii].set_state(
                                    DeviceState.disconnected
                                )

                                num_retries = 0
                                break
                num_retries -= 1
            else:
                num_retries -= 1
                TMSiLogger().warning(
                    "Trying to open connection to device. Number of retries left: "
                    + str(num_retries)
                )

    @LogPerformances
    def download_file_from_device(self, file_id: int, filename: str = None):
        """Creates a data stream to download the file from the device.

        :param file_id: id of the file to download.
        :type file_id: int
        :param verbosity: if True, writes information about the download stream, defaults to True.
        :type verbosity: bool, optional
        :param filename: filename where to write the impedance report (if available), defaults to None
        :type filename: str, optional
        :raises TMSiError: TMSiErrorCode.file_writer_error if impedance report download fails.
        :raises TMSiError: TMSiErrorCode.device_error if the file download fails.
        :raises TMSiError: TMSiErrorCode.api_invalid_command if already sampling.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        header, metadata = self.get_device_card_file_info(file_id)
        n_of_samples = metadata.NumberOfSamples
        self.start_download_file(file_id, filename, n_of_samples)

        while True:
            percentage = self.__measurement.get_download_percentage()
            if percentage >= 100:
                break
            if self.__measurement.is_timeout():
                break
            time.sleep(0.1)

        self.stop_download_file()

    @LogPerformances
    def export_configuration(self, filename: str):
        """Exports the current configuration to an xml file.

        :param filename: name of the file where to export configuration.
        :type filename: str
        :raises TMSiError: TMSiErrorCode.file_writer_error if export configuration fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        if self.__info.get_state() == DeviceState.connected:
            if self.__config.export_to_xml(filename):
                return
            else:
                raise TMSiError(error_code=TMSiErrorCode.file_writer_error)
        raise TMSiError(error_code=TMSiErrorCode.device_not_connected)

    @LogPerformances
    def import_configuration(self, filename: str):
        """Imports the file configuration to the device.

        :param filename: name of the file where to export configuration.
        :type filename: str
        :raises TMSiError: TMSiErrorCode.general_error if import configuration fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        if self.__info.get_state() == DeviceState.connected:
            if self.__config.import_from_xml(filename):
                self.__set_device_channel_config(
                    [
                        ch.get_channel_name()
                        for ch in self.__config.get_channels()
                    ],
                    [i for i in range(len(self.__config.get_channels()))],
                )
                self.__set_device_reference_config(
                    [ch.is_reference() for ch in self.__config.get_channels()],
                    [i for i in range(len(self.__config.get_channels()))],
                )
                self.__set_device_sampling_config(
                    frequency=TMSiBaseSampleRate(
                        self.__config.get_sample_rate()
                    ),
                    impedance_limit=self.__config.get_impedance_limit(),
                    live_impedance=TMSiLiveImpedance(
                        self.__config.get_live_impedance()
                    ),
                )
                self.__load_config_from_device()
                return
            else:
                raise TMSiError(error_code=TMSiErrorCode.general_error)
        raise TMSiError(error_code=TMSiErrorCode.device_not_connected)

    @LogPerformances
    def get_card_recording_config(self) -> TMSiDevCardRecCfg:
        """Gets configuration for card recording.

        :raises TMSiError: TMSiErrorCode.device_error if get configuration fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: Configuration of the card recording.
        :rtype: TMSiDevCardRecCfg
        """
        if self.__info.get_state() == DeviceState.connected:
            return self.__get_device_card_recording_config()
        raise TMSiError(error_code=TMSiErrorCode.device_not_connected)

    @LogPerformances
    def get_card_status(self) -> TMSiDevCardStatus:
        """Gets card status.

        :raises TMSiError: TMSiErrorCode.device_error if get card status fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: Configuration of the card recording.
        :rtype: TMSiDevCardStatus
        """
        if self.__info.get_state() == DeviceState.connected:
            return self.__get_device_card_status()
        raise TMSiError(error_code=TMSiErrorCode.device_not_connected)

    @LogPerformances
    def get_device_active_channels(self):
        """Gets the list of active channels.

        :raises TMSiError: TMSiErrorCode.device_error if get channels from the device fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: The list of channels
        :rtype: list[ApexChannel]
        """
        return self.get_device_channels()

    @LogPerformances
    def get_device_card_file_info(
        self, file_id: int
    ) -> tuple[TMSiFileMetadataHeader, TMSiDevCardFileDetails]:
        """Gets the information of the file on the device's card.

        :param file_id: Id of the file to be investigated.
        :type file_id: int
        :raises TMSiError: TMSiErrorCode.device_error if get card info fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: A tuple with file metadata and file details.
        :rtype: tuple[TMSiFileMetadataHeader, TMSiDevCardFileDetails]
        """
        if self.__info.get_state() == DeviceState.connected:
            metadata, _ = self.__get_device_card_file_metadata(file_id)
            header = self.__get_device_card_file_metadata_header(file_id)
            return (header, metadata)
        raise TMSiError(error_code=TMSiErrorCode.device_not_connected)

    @LogPerformances
    def get_device_card_file_list(self) -> list[TMSiDevCardFileInfo]:
        """Gets the list of files available on the device's card.

        :raises TMSiError: TMSiErrorCode.device_error if get card file list fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: A list of file info.
        :rtype: list[TMSiDevCardFileInfo]
        """
        if self.__info.get_state() == DeviceState.connected:
            return self.__get_device_card_file_list()
        raise TMSiError(error_code=TMSiErrorCode.device_not_connected)

    @LogPerformances
    def get_device_channels(self) -> list[ApexChannel]:
        """Gets the list of channels.

        :raises TMSiError: TMSiErrorCode.device_error if get channels from the device fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: The list of channels
        :rtype: list[ApexChannel]
        """
        if self.__info.get_state() == DeviceState.sampling:
            return self.__config.get_channels()
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code=TMSiErrorCode.device_not_connected)
        (
            device_channel_metadata,
            device_cycling_state_metadata,
        ) = self.__get_device_sample_metadata()
        (
            device_channel_name_list,
            device_channel_alt_name_list,
        ) = self.__get_device_channel_config()
        device_reference_config = self.__get_device_reference_config()
        channels = []
        for i in range(self.get_num_channels()):
            channel = ApexChannel()
            channel.set_device_channel_information(device_channel_metadata[i])
            channel.set_device_channel_names(
                device_channel_name_list[i].ChanName.decode("windows-1252"),
                device_channel_alt_name_list[i].AltChanName.decode(
                    "windows-1252"
                ),
            )
            if i < len(device_reference_config):
                channel.set_device_reference(
                    device_reference_config[i].ChanRefStatus
                )
            else:
                channel.set_device_reference(None)
            channels.append(channel)
        self.__config.set_channels(channels)
        self.__get_device_impedance_metadata()
        return self.__config.get_channels()

    @LogPerformances
    def get_device_data(
        self,
        POINTER_received_data_array: pointer,
        buffer_size: int,
        POINTER_num_of_sets: pointer,
        POINTER_data_type: pointer,
    ) -> TMSiDeviceRetVal:
        """Gets data from the device during sampling.

        :param POINTER_received_data_array: array that will contain the received data.
        :type POINTER_received_data_array: pointer(array[c_float])
        :param buffer_size: maximum size of the buffer.
        :type buffer_size: int
        :param POINTER_num_of_sets: number of sets of data received.
        :type POINTER_num_of_sets: pointer(c_uint)
        :param POINTER_data_type: type of data received.
        :type POINTER_data_type: pointer(c_int)
        :raises TMSiError: TMSiErrorCode.api_invalid_command if device is notin  samplin modeg
        :return: return value of the call
        :rtype: TMSiDeviceRetVal
        """
        if self.__info.get_state() != DeviceState.sampling:
            raise TMSiError(error_code=TMSiErrorCode.api_invalid_command)
        return TMSiGetDeviceData(
            self.__device_handle,
            POINTER_received_data_array,
            buffer_size,
            POINTER_num_of_sets,
            POINTER_data_type,
        )

    @LogPerformances
    def get_device_handle_value(self) -> int:
        """Returns the value of the device handle.

        :return: Device handle.
        :rtype: int
        """
        return self.__device_handle.value

    @LogPerformances
    def get_device_impedance_channels(self) -> list[ApexImpedanceChannel]:
        """Gets the list of impedance channels.

        :raises TMSiError: TMSiErrorCode.device_error if get channels from the device fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: The list of channels
        :rtype: list[ApexChannel]
        """
        self.get_device_channels()
        return self.__config.get_impedance_channels()

    @LogPerformances
    def get_device_impedance_data(
        self,
        POINTER_received_data_array: pointer,
        buffer_size: int,
        POINTER_num_of_sets: pointer,
    ) -> TMSiDeviceRetVal:
        """Gets impedance data from the device during sampling.

        :param POINTER_received_data_array: array that will contain the received data.
        :type POINTER_received_data_array: pointer(array[TMSiImpedanceSample])
        :param buffer_size: maximum size of the buffer.
        :type buffer_size: int
        :param POINTER_num_of_sets: number of sets of data received.
        :type POINTER_num_of_sets: pointer
        :raises TMSiError: TMSiErrorCode.api_invalid_command if device is notin  samplin modeg
        :return: return value of the call
        :rtype: TMSiDeviceRetVal
        """
        if self.__info.get_state() != DeviceState.sampling:
            raise TMSiError(error_code=TMSiErrorCode.api_invalid_command)
        return TMSiGetDeviceImpedanceData(
            self.__device_handle,
            POINTER_received_data_array,
            buffer_size,
            POINTER_num_of_sets,
        )

    @LogPerformances
    def get_device_info_report(self) -> TMSiDevInfoReport:
        """Gets the report with device information.

        :raises TMSiError: TMSiErrorCode.device_error if get device info from the device fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: the device info report.
        :rtype: TMSiDevInfoReport
        """
        if self.__info.get_state() == DeviceState.connected:
            device_info_report = self.__get_device_info_report()
            self.__info.set_device_info_report(device_info_report)
            return device_info_report
        raise TMSiError(error_code=TMSiErrorCode.device_not_connected)

    @LogPerformances
    def get_device_interface_status(
        self, interface=DeviceInterfaceType.bluetooth
    ) -> TMSiInterfaceStatus:
        """Gets the device interface status.

        :param interface: interface to check, defaults to DeviceInterfaceType.bluetooth
        :type interface: DeviceInterfaceType, optional
        :raises TMSiError: TMSiErrorCode.device_error if get device interface status from the device fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: interface status of the device.
        :rtype: TMSiInterfaceStatus
        """
        if self.__info.get_state() == DeviceState.connected:
            device_interface_status = self.__get_device_interface_status(
                interface
            )
            return device_interface_status
        raise TMSiError(error_code=TMSiErrorCode.device_not_connected)

    @LogPerformances
    def get_device_list(
        dr_interface: DeviceInterfaceType,
    ) -> list["ApexDevice"]:
        """Gets the list of available devices.

        :param dr_interface: interface to check.
        :type dr_interface: DeviceInterfaceType
        :return: a list of available devices on the requested interface.
        :rtype: list[ApexDevice]
        """
        device_list = []
        for idx in range(ApexDevice.__MAX_NUM_DEVICES):
            if (
                ApexDevice.__device_info_list[idx].get_id()
                != ApexConst.TMSI_DEVICE_ID_NONE
                and ApexDevice.__device_info_list[idx].get_dr_interface()
                == dr_interface
            ):
                device_list.append(
                    ApexDevice(
                        dongle_serial_number=ApexDevice.__device_info_list[
                            idx
                        ].get_dongle_serial_number(),
                        serial_number=ApexDevice.__device_info_list[
                            idx
                        ].get_dr_serial_number(),
                        dr_interface=dr_interface,
                        idx=ApexDevice.__device_info_list[idx].get_id(),
                        pairing_status=ApexDevice.__device_info_list[
                            idx
                        ].get_pairing_status(),
                    )
                )
        return device_list

    @LogPerformances
    def get_device_power_status(self) -> TMSiDevPowerStatus:
        """Returns the power status of the device

        :raises TMSiError: TMSiErrorCode.device_error if get device power status from the device fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: device power status.
        :rtype: TMSiDevPowerStatus
        """
        if self.__info.get_state() == DeviceState.connected:
            return self.__get_device_power_status()
        raise TMSiError(error_code=TMSiErrorCode.device_not_connected)

    @LogPerformances
    def get_device_sampling_config(self) -> TMSiDevSamplingCfg:
        """Gets the sampling configuration of the device.

        :raises TMSiError: TMSiErrorCode.device_error if get device sampling configuration from the device fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: device sampling configuration
        :rtype: TMSiDevSamplingCfg
        """
        if self.__info.get_state() == DeviceState.connected:
            device_sampling_config = self.__get_device_sampling_config()
            self.__config.set_device_sampling_config(device_sampling_config)
            return device_sampling_config
        raise TMSiError(error_code=TMSiErrorCode.device_not_connected)

    @LogPerformances
    def get_device_sampling_frequency(self):
        """Gets the sampling frequency."""
        return self.__config.get_sampling_frequency()

    @LogPerformances
    def get_device_serial_number(self) -> int:
        """Gets the serial number of the device.

        :return: serial number of the device.
        :rtype: int
        """
        return self.__info.get_dr_serial_number()

    @LogPerformances
    def get_device_state(self) -> DeviceState:
        """Gets the state of the device.

        :return: the device state.
        :rtype: DeviceState
        """
        return self.__info.get_state()

    @LogPerformances
    def get_device_time(self) -> datetime.datetime:
        """Gets the time on the device.

        :raises TMSiError: TMSiErrorCode.device_error if get time from the device fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: returns the datetime on the device.
        :rtype: datetime.datetime
        """
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code=TMSiErrorCode.device_not_connected)
        return self.__get_device_rtc()

    @LogPerformances
    def get_device_type(self) -> str:
        """Returns the device type.

        :return: the device type.
        :rtype: str
        """
        return ApexDevice.__DEVICE_TYPE

    @LogPerformances
    def get_dongle_list() -> list[ApexDongle]:
        """Returns the list of available dongles.

        :return: list of available dongles.
        :rtype: list[ApexDongle]
        """
        dongle_list = []
        for idx in range(ApexDevice.__MAX_NUM_DONGLES):
            if (
                ApexDevice.__dongle_info_list[idx].TMSiDongleID
                != ApexConst.TMSI_DONGLE_ID_NONE
            ):
                dongle_list.append(
                    ApexDongle(
                        ApexDevice.__dongle_info_list[idx].TMSiDongleID,
                        ApexDevice.__dongle_info_list[idx].SerialNumber,
                    )
                )
        return dongle_list

    @LogPerformances
    def get_dongle_serial_number(self) -> int:
        """Gets the serial number of the dongle.

        :return: serial number of the device.
        :rtype: int
        """
        return self.__info.get_dongle_serial_number()

    @LogPerformances
    def get_downloaded_percentage(self) -> int:
        """Gets the percentage of file downloaded.

        :return: percentage of file downloaded.
        :rtype: int
        """
        return self.__measurement.get_download_percentage()

    @LogPerformances
    def get_driver_version() -> tuple[str, str]:
        """Gets the version of the DLL and USB drivers.

        :raises TMSiError: TMSiErrorCode.device_error if it fails.
        :return: tuple(dll_version, usb_version)
        :rtype: tuple(str, str)
        """
        version = TMSiDeviceDriverVersionInfo()
        __last_error = TMSiGetDeviceDriverVersionInfo(pointer(version))
        if __last_error != TMSiDeviceRetVal.TMSiStatusOK:
            raise TMSiError(
                error_code=TMSiErrorCode.device_error,
                dll_error_code=__last_error,
            )
        return version

    @LogPerformances
    def get_dr_interface(self) -> DeviceInterfaceType:
        """Returns the interface of the device.

        :return: the interface of the device.
        :rtype: DeviceInterfaceType
        """
        return self.__info.get_dr_interface()

    @LogPerformances
    def get_event(POINTER_event):
        """Gets the last event available

        :param POINTER_event: _description_
        :type POINTER_event: _type_
        :return: _description_
        :rtype: _type_
        """
        __last_error = TMSiGetEvent(POINTER_event)
        if __last_error != TMSiDeviceRetVal.TMSiStatusOK:
            raise TMSiError(
                error_code=TMSiErrorCode.device_error,
                dll_error_code=__last_error,
            )

    @LogPerformances
    def get_event_buffer(POINTER_num_found_events):
        """Gets the available events in the buffer

        :return: TMSiDeviceRetVal.TMSiStatusOK
        :rtype: TMSiDeviceRetVal
        """
        return TMSiGetEventBuffered(POINTER_num_found_events)

    @LogPerformances
    def get_id(self) -> int:
        """Gets the device id.

        :return: the device id.
        :rtype: int
        """
        return self.__info.get_id()

    @LogPerformances
    def get_live_impedance(self) -> bool:
        """Returns if live impedance is enabled.

        :return: True if enabled, False if disabled.
        :rtype: bool
        """
        return self.__config.get_live_impedance()

    @LogPerformances
    def get_num_channels(self) -> int:
        """Returns the number of channels of the device.

        :return: number of channels of the device.
        :rtype: int
        """
        return self.__info.get_num_channels()

    @LogPerformances
    def get_num_impedance_channels(self) -> int:
        """Returns the number of impedance channels of the device.

        :return: number of impedance channels of the device.
        :rtype: int
        """
        return self.__info.get_num_impedance_channels()

    @LogPerformances
    def get_sdk() -> CDLL:
        """Gets the handle of the communication library

        :return: the handle of the communication library.
        :rtype: CDLL
        """
        if ApexDevice.__apex_sdk is None:
            ApexDevice.__initialize()
        return ApexDevice.__apex_sdk

    @LogPerformances
    def open(self, dongle_id=ApexConst.TMSI_DONGLE_ID_NONE):
        """Opens the connection with the device.

        :param dongle_id: Id of the dongle to use for the connection. Defaults to ApexConst.TMSI_DONGLE_ID_NONE
        :type dongle_id: int, optional
        :raises TMSiError: TMSiErrorCode.device_error if get time from the device fails.
        :raises TMSiError: TMSiErrorCode.no_devices_found if device not found.
        """
        self.__info.set_dongle_id(dongle_id)
        if self.__info.get_id() != ApexConst.TMSI_DEVICE_ID_NONE:
            self.__last_error_code = TMSiOpenInterface(
                pointer(self.__device_handle),
                dongle_id,
                self.__info.get_id(),
                self.__info.get_dr_interface().value,
            )
            if (
                self.__last_error_code
                == TMSiDeviceRetVal.TMSiStatusDrInterfaceAlreadyOpen
            ):
                # The found device is available but in it's open-state: Close and re-open the connection
                self.__last_error_code = TMSiCloseInterface(
                    self.__device_handle, self.__info.get_dr_interface().value
                )
                self.__last_error_code = TMSiOpenInterface(
                    pointer(self.__device_handle),
                    dongle_id,
                    self.__info.get_id(),
                    self.__info.get_dr_interface().value,
                )

            if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
                # The device is opened succesfully. Update the device information.
                self.__info.set_state(DeviceState.connected)

                # Read the device's configuration
                self.__load_config_from_device()

            else:
                raise TMSiError(
                    error_code=TMSiErrorCode.device_error,
                    dll_error_code=self.__last_error_code,
                )
        else:
            raise TMSiError(error_code=TMSiErrorCode.no_devices_found)

    @LogPerformances
    def pair_device(dongle_id: int, device_id: int) -> TMSiDeviceRetVal:
        """Pairs the device with the dongle

        :param dongle_id: Id of the dongle
        :type dongle_id: int
        :param device_id: Id of the device
        :type device_id: int
        :return: return value of the call
        :rtype: TMSiDeviceRetVal
        """
        ret = TMSiPairDevice(dongle_id, device_id)
        if ret == TMSiDeviceRetVal.TMSiStatusDllCommandInProgress:
            TMSiLogger().info("Pairing ongoing")
        elif ret == TMSiDeviceRetVal.TMSiStatusDllDeviceAlreadyPaired:
            TMSiLogger().warning("Device already paired")
        else:
            TMSiLogger().warning("Pairing process failed")
        return ret

    @LogPerformances
    def reset_device_card(self):
        """Resets the memory card of the device.

        :raises TMSiError: TMSiErrorCode.device_error if reset card fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code=TMSiErrorCode.device_not_connected)
        self.__reset_device_card()

    @LogPerformances
    def reset_device_data_buffer(self):
        """Resets the incoming buffer.

        :raises TMSiError: TMSiErrorCode.api_invalid_command if device is notin  samplin modeg
        :raises TMSiError: TMSiErrorCode.device_error if reset fails
        """
        if self.__info.get_state() != DeviceState.sampling:
            raise TMSiError(TMSiErrorCode.api_invalid_command)
        self.__last_error_code = TMSiResetDeviceDataBuffer(self.__device_handle)
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def reset_device_event_buffer():
        """Resets the incoming event buffer.

        :raises TMSiError: TMSiErrorCode.device_error if reset fails
        """
        __last_error_code = TMSiResetEventBuffer()
        if __last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return
        else:
            raise TMSiError(TMSiErrorCode.device_error, __last_error_code)

    @LogPerformances
    def reset_to_factory_default(self):
        """Resets the device to default configuration.

        :raises TMSiError: TMSiErrorCode.device_error if reset fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code=TMSiErrorCode.device_not_connected)
        self.__set_device_factory_default()
        self.__load_config_from_device()

    @LogPerformances
    def set_card_recording_config(
        self, config: TMSiDevCardRecCfg
    ) -> TMSiDevCardRecCfg:
        """Sets the configuration for recording on card.

        :param config: configuration to be set.
        :type config: TMSiDevCardRecCfg
        :raises TMSiError: TMSiErrorCode.device_error if set configuration fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: the new available configuration for recording on card.
        :rtype: TMSiDevCardRecCfg
        """
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code=TMSiErrorCode.device_not_connected)
        self.__set_device_card_recording_config(config)
        return self.__get_device_card_recording_config()

    @LogPerformances
    def set_device_channel_names(
        self, names: list[str], indices: list[int]
    ) -> list[ApexChannel]:
        """Sets the device channel names

        :param names: names to be set.
        :type names: list[str]
        :param indices: index of the channels to edit.
        :type indices: list[int]
        :raises TMSiError: TMSiErrorCode.device_error if set names fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :raises TypeError: if names is not only strings
        :raises TypeError: if indices is not only integers.
        :return: list of new channels.
        :rtype: list[ApexChannel]
        """
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code=TMSiErrorCode.device_not_connected)
        for name in names:
            if not isinstance(name, str):
                raise TypeError("names must be strings")
        for index in indices:
            if not isinstance(index, int):
                raise TypeError("indices must be integers")
        self.__set_device_channel_config(names, indices)
        return self.get_device_channels()

    @LogPerformances
    def set_device_download_file_request(
        self, file_request: TMSiDevSetCardFileReq
    ):
        """Sets the download file request to start or stop the download stream.

        :param file_request: file request to start or stop.
        :type file_request: TMSiDevSetCardFileReq
        :raises TMSiError: TMSiErrorCode.device_error if set impedance request fails.
        :raises TMSiError: TMSiErrorCode.api_invalid_command if device is not in sampling mode.
        """
        if self.__info.get_state() != DeviceState.sampling:
            raise TMSiError(error_code=TMSiErrorCode.api_invalid_command)
        self.__last_error_code = TMSiSetDeviceCardFileRequest(
            self.__device_handle, pointer(file_request)
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def set_device_impedance_request(
        self, measurement_request: TMSiDevImpedanceRequest
    ):
        """Sets the impedance request to start or stop the acquisition.

        :param measurement_request: measurement request to start or stop.
        :type measurement_request: TMSiSetDeviceImpedanceRequest
        :raises TMSiError: TMSiErrorCode.device_error if set impedance request fails.
        :raises TMSiError: TMSiErrorCode.api_invalid_command if device is not in sampling mode.
        """
        if self.__info.get_state() != DeviceState.sampling:
            raise TMSiError(error_code=TMSiErrorCode.api_invalid_command)
        self.__last_error_code = TMSiSetDeviceImpedanceRequest(
            self.__device_handle, pointer(measurement_request)
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def set_device_interface(
        self,
        device_interface=DeviceInterfaceType.bluetooth,
        control=TMSiControl.ControlEnabled,
    ) -> TMSiDevCardStatus:
        """Set the device interface.

        :param device_interface: the interface to set, defaults to DeviceInterfaceType.bluetooth
        :type device_interface: DeviceInterfaceType, optional
        :param control: enable or disable interface, defaults to TMSiControl.ControlEnabled
        :type control: TMSiControl, optional
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: the new status of the device interface.
        :rtype: TMSiDevCardStatus
        """
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code=TMSiErrorCode.device_not_connected)
        self.__set_device_interface_config(
            device_interface=device_interface, control=control
        )
        return self.get_device_interface_status(device_interface)

    @LogPerformances
    def set_device_references(
        self, list_references: list[int], list_indices: list[int]
    ):
        """Sets the channels to be used as reference.

        :param list_references: list of reference values.
        :type list_references: list[int]
        :param list_indices: list of indices to be edited.
        :type list_indices: list[int]
        :raises TMSiError: TMSiErrorCode.device_error if set references fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :raises TypeError: if references is not only integers
        :raises TypeError: if indices is not only integers.
        :return: list of new channels.
        :rtype: list[ApexChannel]
        """
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code=TMSiErrorCode.device_not_connected)
        for index in list_indices:
            if not isinstance(index, int):
                raise TypeError("indices must be integers")
        for reference in list_references:
            if not isinstance(reference, int):
                raise TypeError("references must be integers")
        self.__set_device_reference_config(
            list_references=list_references, list_indices=list_indices
        )
        return self.get_device_channels()

    @LogPerformances
    def set_device_sampling_config(
        self,
        sampling_frequency=TMSiBaseSampleRate.Decimal,
        live_impedance=TMSiLiveImpedance.On,
        impedance_limit=0,
    ) -> TMSiDevSamplingCfg:
        """Sets the sampling configuration of the device.

        :param sampling_frequency: sample rate to set, defaults to TMSiBaseSampleRate.Decimal
        :type sampling_frequency: TMSiBaseSampleRate, optional
        :param live_impedance: enable or disable live impedance, defaults to TMSiLiveImpedance.On
        :type live_impedance: TMSiLiveImpedance, optional
        :param impedance_limit: kOhms value of the impedance limit, defaults to 0
        :type impedance_limit: int, optional
        :raises TMSiError: TMSiErrorCode.device_error if set sampling configuration fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: the new sampling configuration
        :rtype: TMSiDevSamplingCfg
        """
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code=TMSiErrorCode.device_not_connected)
        self.__set_device_sampling_config(
            frequency=sampling_frequency,
            live_impedance=live_impedance,
            impedance_limit=impedance_limit,
        )
        return self.get_device_sampling_config()

    @LogPerformances
    def set_device_sampling_request(
        self, measurement_request: TMSiDevSampleRequest
    ):
        """Sets the sampling request to start or stop the acquisition.

        :param measurement_request: measurement request to configure the acquisition.
        :type measurement_request: TMSiDevSampleRequest
        :raises TMSiError: TMSiErrorCode.device_error if set sampling request fails.
        :raises TMSiError: TMSiErrorCode.api_invalid_command if device is not in sampling mode.
        """
        if self.__info.get_state() != DeviceState.sampling:
            raise TMSiError(error_code=TMSiErrorCode.api_invalid_command)
        self.__last_error_code = TMSiSetDeviceSamplingRequest(
            self.__device_handle, pointer(measurement_request)
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def set_device_time(self, datetime: datetime.datetime):
        """Sets the datetime of the device.

        :param datetime: datetime to set on the device.
        :type datetime: datetime.datetime
        :raises TMSiError: TMSiErrorCode.device_error if set time fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code=TMSiErrorCode.device_not_connected)
        self.__set_device_rtc(datetime)

    @LogPerformances
    def start_download_file(
        self, file_id: int, filename: str = None, n_of_samples: int = None
    ):
        """Starts the download of the file requested.

        :param file_id: id of the file to download.
        :type file_id: int
        :param filename: filename where to write the impedance report (if available), defaults to None
        :type filename: str, optional
        :raises TMSiError: TMSiErrorCode.file_writer_error if impedance report download fails.
        :raises TMSiError: TMSiErrorCode.api_invalid_command if already sampling.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        if self.__info.get_state() == DeviceState.sampling:
            raise TMSiError(TMSiErrorCode.api_invalid_command)
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(TMSiErrorCode.device_not_connected)
        if filename is not None:
            self.__download_impedance_report(file_id, filename)
        self.__measurement = MeasurementType.APEX_DOWNLOAD(
            self, file_id, n_of_samples
        )
        self.__info.set_state(DeviceState.sampling)
        self.__measurement.start()

    @LogPerformances
    def start_measurement(
        self, measurement_type: MeasurementType, thread_refresh=None
    ):
        """Starts the measurement requested.

        :param measurement_type: measurement to start
        :type measurement_type: MeasurementType
        :param thread_refresh: refresh time for sampling and conversion threads, defaults to None.
        :type thread_refresh: float, optional.
        :raises TMSiError: TMSiErrorCode.api_invalid_command if already sampling.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        if self.__info.get_state() == DeviceState.sampling:
            raise TMSiError(TMSiErrorCode.api_invalid_command)
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code=TMSiErrorCode.device_not_connected)
        self.__measurement = measurement_type(self)
        if thread_refresh is not None:
            self.__measurement.set_sampling_pause(thread_refresh)
            self.__measurement.set_conversion_pause(thread_refresh)
        self.__info.set_state(DeviceState.sampling)
        self.__measurement.start()

    @LogPerformances
    def stop_download_file(self):
        """Stops the download of the file.

        :raises TMSiError: TMSiErrorCode.api_invalid_command if not sampling.
        """
        if self.__info.get_state() != DeviceState.sampling:
            raise TMSiError(error_code=TMSiErrorCode.api_invalid_command)
        self.__measurement.stop()
        self.__info.set_state(DeviceState.connected)

    @LogPerformances
    def stop_measurement(self):
        """Stops the measurement requested.

        :param file_id: id of the file to download.
        :type file_id: int
        :raises TMSiError: TMSiErrorCode.api_invalid_command if already sampling.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        if self.__info.get_state() != DeviceState.sampling:
            raise TMSiError(error_code=TMSiErrorCode.api_invalid_command)
        self.__measurement.stop()
        self.__info.set_state(DeviceState.connected)

    @LogPerformances
    def __initialize():
        try:
            ApexDevice.__apex_sdk = ApexSDK
            for i in range(ApexDevice.__MAX_NUM_DEVICES):
                ApexDevice.__device_info_list.append(ApexInfo())
            for i in range(ApexDevice.__MAX_NUM_DONGLES):
                ApexDevice.__dongle_info_list.append(DongleInfo())
        except:
            ApexDevice.__apex_sdk = None
            raise TMSiError(error_code=TMSiErrorCode.api_no_driver)

    @LogPerformances
    def __download_impedance_report(self, file_id, filename):
        (
            card_file_details,
            impedance_report_list,
        ) = self.__get_device_card_file_metadata(file_id)
        try:
            with open("{}.txt".format(filename), "w") as f:
                f.write("{}\n\n".format(card_file_details.RecFileName.decode()))
                f.write("Idx\t\tName\t\tRe\t\t\tIm\n")
                f.write(
                    "\n".join(
                        [
                            "{}\t\t{}\t\t{} kOhm\t\t{} nF".format(
                                i.ChanIdx,
                                i.ChanName.decode(),
                                i.ImpedanceRe,
                                i.ImpedanceIm,
                            )
                            for i in impedance_report_list
                        ]
                    )
                )
        except Exception as e:
            raise TMSiError(TMSiErrorCode.file_writer_error)

    @LogPerformances
    def __get_device_card_file_list(self):
        file_list = (2000 * TMSiDevCardFileInfo)()
        file_number = (c_uint)(0)
        self.__last_error_code = TMSiGetDeviceCardFileList(
            self.__device_handle,
            pointer(file_list),
            len(file_list),
            pointer(file_number),
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return_list = []
            for i in range(file_number.value):
                return_list.append(file_list[i])
            return return_list
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __get_device_card_file_metadata(self, file_id):
        RecFileID = file_id
        DevCardFileDetails = TMSiDevCardFileDetails()
        ChannelMetadataList = (self.get_num_channels() * TMSiChannelMetadata)()
        ChannelMetadataListLen = self.get_num_channels()
        RetChannelMetadataListLen = (c_uint)(0)
        CyclingStateMetadataList = (
            self.get_num_channels() * TMSiCyclingStateMetadata
        )()
        CyclingStateMetadataListLen = self.get_num_channels()
        RetCyclingStatMetadataListLen = (c_uint)(0)
        ImpReportMetadata = TMSiDevImpReportMetadata()
        ImpedanceReportList = (self.get_num_channels() * TMSiDevImpReport)()
        ImpedanceReportListLen = self.get_num_channels()
        RetImpedanceReportListLen = (c_uint)(0)
        self.__last_error_code = TMSiGetDeviceCardFileMetadata(
            self.__device_handle,
            RecFileID,
            pointer(DevCardFileDetails),
            pointer(ChannelMetadataList),
            ChannelMetadataListLen,
            pointer(RetChannelMetadataListLen),
            pointer(CyclingStateMetadataList),
            CyclingStateMetadataListLen,
            pointer(RetCyclingStatMetadataListLen),
            pointer(ImpReportMetadata),
            pointer(ImpedanceReportList),
            ImpedanceReportListLen,
            pointer(RetImpedanceReportListLen),
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return (
                DevCardFileDetails,
                ImpedanceReportList[0 : RetImpedanceReportListLen.value],
            )
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __get_device_card_file_metadata_header(self, file_id):
        metadata = TMSiFileMetadataHeader()
        self.__last_error_code = TMSiGetDeviceCardFileMetadataHeader(
            self.__device_handle, file_id, pointer(metadata)
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return metadata
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __get_device_card_recording_config(self):
        config = TMSiDevCardRecCfg()
        self.__last_error_code = TMSiGetDeviceCardRecordingConfig(
            self.__device_handle, pointer(config)
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return config
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __get_device_card_status(self):
        card_status = TMSiDevCardStatus()
        self.__last_error_code = TMSiGetDeviceCardStatus(
            self.__device_handle, pointer(card_status)
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return card_status
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __get_device_channel_config(self):
        device_channel_alt_name_list = (
            TMSiDevAltChName * self.get_num_channels()
        )()
        device_channel_name_list = (TMSiDevChName * self.get_num_channels())()
        num_returned_items = (c_uint)(0)
        self.__last_error_code = TMSiGetDeviceChannelConfig(
            self.__device_handle,
            pointer(device_channel_name_list),
            pointer(device_channel_alt_name_list),
            self.get_num_channels(),
            pointer(num_returned_items),
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return device_channel_name_list, device_channel_alt_name_list
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __get_device_impedance_metadata(self):
        num_returned_items = (c_uint)(0)
        device_impedance_metadata_list = (
            TMSiImpedanceMetadata * self.get_num_impedance_channels()
        )()
        TMSiGetDeviceImpedanceMetadata(
            self.__device_handle,
            pointer(device_impedance_metadata_list),
            self.get_num_impedance_channels(),
            pointer(num_returned_items),
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            impedance_metadata_list = []
            impedance_channels = []
            for i in range(num_returned_items.value):
                impedance_metadata_list.append(
                    device_impedance_metadata_list[i]
                )
                impedance_channels.append(
                    ApexImpedanceChannel(device_impedance_metadata_list[i])
                )
            self.__config.set_impedance_channels(impedance_channels)
            return impedance_metadata_list
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __get_device_info_report(self):
        device_info_report = TMSiDevInfoReport()
        self.__last_error_code = TMSiGetDeviceInfo(
            self.__device_handle, pointer(device_info_report)
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return device_info_report
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __get_device_interface_status(self, interface):
        interface_status = TMSiInterfaceStatus()
        self.__last_error_code = TMSiGetDeviceInterfaceStatus(
            self.__device_handle, interface.value, pointer(interface_status)
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return interface_status
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __get_device_power_status(self):
        power_status = TMSiDevPowerStatus()
        self.__last_error_code = TMSiGetDevicePowerStatus(
            self.__device_handle, pointer(power_status)
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return power_status
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __get_device_reference_config(self):
        device_channel_reference_list = (
            TMSiDevChanRef * self.get_num_channels()
        )()
        len_channel_reference_list = (c_uint)(0)
        self.__last_error_code = TMSiGetDeviceReferenceConfig(
            self.__device_handle,
            pointer(device_channel_reference_list),
            self.get_num_channels(),
            pointer(len_channel_reference_list),
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            device_channel_reference_list_to_return = (
                TMSiDevChanRef * len_channel_reference_list.value
            )()
            for i in range(len_channel_reference_list.value):
                device_channel_reference_list_to_return[
                    i
                ] = device_channel_reference_list[i]
            return device_channel_reference_list_to_return
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __get_device_rtc(self):
        dev_time = TMSiTime()
        self.__last_error_code = TMSiGetDeviceRTC(
            self.__device_handle, pointer(dev_time)
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return datetime.datetime(
                dev_time.Year + 1900,
                dev_time.Month + 1,
                dev_time.DayOfMonth,
                dev_time.Hours,
                dev_time.Minutes,
                dev_time.Seconds,
            )
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __get_device_sample_metadata_header(self):
        device_sampling_metadata_header = TMSiSampleMetadataHeader()
        self.__last_error_code = TMSiGetDeviceSampleMetadataHeader(
            self.__device_handle,
            self.__info.get_dr_interface().value,
            pointer(device_sampling_metadata_header),
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return device_sampling_metadata_header
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __get_device_sample_metadata(self):
        device_channel_metadata = (
            TMSiChannelMetadata * self.get_num_channels()
        )()
        device_cycling_state_metadata = (
            TMSiCyclingStateMetadata * self.get_num_channels()
        )()
        num_returned_channels = (c_uint)(0)
        num_returned_cycling_states = (c_uint)(0)
        TMSiGetDeviceSampleMetadata(
            self.__device_handle,
            self.__info.get_dr_interface().value,
            pointer(device_channel_metadata),
            self.get_num_channels(),
            pointer(num_returned_channels),
            pointer(device_cycling_state_metadata),
            self.get_num_channels(),
            pointer(num_returned_cycling_states),
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return_channels = []
            return_cycling_states = []
            for i in range(num_returned_channels.value):
                return_channels.append(device_channel_metadata[i])
            for i in range(num_returned_cycling_states.value):
                return_cycling_states.append(device_cycling_state_metadata[i])
            return return_channels, return_cycling_states
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __get_device_sampling_config(self):
        device_sampling_config = TMSiDevSamplingCfg()
        self.__last_error_code = TMSiGetDeviceSamplingConfig(
            self.__device_handle, pointer(device_sampling_config)
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return device_sampling_config
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __load_config_from_device(self):
        self.get_device_info_report()
        self.get_device_sampling_config()
        device_sampling_metadata_header = (
            self.__get_device_sample_metadata_header()
        )
        self.__config.set_sampling_frequency(
            device_sampling_metadata_header.BaseFS
        )
        self.get_device_channels()

    @LogPerformances
    def __reset_device_card(self):
        self.__last_error_code = TMSiResetDeviceCard(self.__device_handle)
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __set_device_factory_default(self):
        self.__last_error_code = TMSiSetDeviceFactoryDefaults(
            self.__device_handle
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __set_device_card_recording_config(self, config):
        self.__last_error_code = TMSiSetDeviceCardRecordingConfig(
            self.__device_handle, pointer(config)
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __set_device_channel_config(self, list_names, list_indices):
        _, device_channel_alt_name_list = self.__get_device_channel_config()
        num_channels = len(list_indices)
        if len(list_indices) > 0:
            for i in range(num_channels):
                converted_str = bytearray(ApexStringLengths.AltChanName.value)
                for character in range(len(list_names[i])):
                    converted_str[character] = ord(list_names[i][character])
                device_channel_alt_name_list[
                    list_indices[i]
                ].AltChanName = bytes(converted_str)
        self.__last_error_code = TMSiSetDeviceChannelConfig(
            self.__device_handle,
            device_channel_alt_name_list,
            len(device_channel_alt_name_list),
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __set_device_interface_config(
        self,
        device_interface=DeviceInterfaceType.bluetooth,
        control=TMSiControl.ControlEnabled,
    ):
        self.__last_error_code = TMSiSetDeviceInterfaceConfig(
            self.__device_handle, device_interface.value, control.value
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __set_device_reference_config(self, list_references, list_indices):
        device_channel_reference_list = self.__get_device_reference_config()
        if len(list_indices) > 0:
            for i in range(len(list_indices)):
                if list_indices[i] < len(device_channel_reference_list):
                    device_channel_reference_list[
                        list_indices[i]
                    ].ChanRefStatus = list_references[i]
        self.__last_error_code = TMSiSetDeviceReferenceConfig(
            self.__device_handle,
            pointer(device_channel_reference_list),
            len(device_channel_reference_list),
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __set_device_rtc(self, dt):
        dev_time = TMSiTime()
        dev_time.Year = dt.year - 1900
        dev_time.Month = dt.month - 1
        dev_time.DayOfMonth = dt.day
        dev_time.Hours = dt.hour
        dev_time.Minutes = dt.minute
        dev_time.Seconds = dt.second
        self.__last_error_code = TMSiSetDeviceRTC(
            self.__device_handle, pointer(dev_time)
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)

    @LogPerformances
    def __set_device_sampling_config(
        self,
        frequency=TMSiBaseSampleRate.Binary,
        live_impedance=TMSiLiveImpedance.On,
        impedance_limit=0,
    ):
        device_sampling_config = TMSiDevSamplingCfg()
        device_sampling_config.BaseSampleRate = frequency.value
        device_sampling_config.ImpedanceLimit = impedance_limit
        device_sampling_config.LiveImpedance = live_impedance.value
        self.__last_error_code = TMSiSetDeviceSamplingConfig(
            self.__device_handle, pointer(device_sampling_config)
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSiStatusOK:
            return
        else:
            raise TMSiError(TMSiErrorCode.device_error, self.__last_error_code)
