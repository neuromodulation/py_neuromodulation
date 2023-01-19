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
 * @file ${saga_device.py} 
 * @brief SAGA Device Interface 
 *
 */


'''

import os

from .saga_types import *
from ...error import TMSiError, TMSiErrorCode
from ...device import Device, DeviceChannel, ChannelType, MeasurementType, \
                      DeviceInfo, DeviceState, DeviceStatus, DeviceSensor, DeviceInterfaceType
if os.getenv("ENVIRONMENT") == "TEST":
    # do not load the dll
    print("Test environment recognized. Dll is not going to be loaded.")
else:
    from .TMSi_Device_API import *

import array
from copy import copy,deepcopy
import datetime
import struct
import threading
import time
import queue
import warnings
import numpy as np

import tkinter as tk
from tkinter import messagebox

from ... import sample_data, sample_data_server

from .xml_saga_config import *

_tmsi_sdk = None
_device_info_list = []
_MAX_NUM_DEVICES = 2
_MAX_NUM_BATTERIES = 2

class SagaDevice(Device):
    """ 'Device' handles the connection to a TMSi Measurement System like the SAGA.

        The Device class interfaces with the measurement system to :
            - open/close a connection to the system
            - configure the system
            - forward the received sample data to Python-clients for display and/or storage.

        Args:
            ds-interface: The interface-type between the PC and the docking-station.
                          This might be 'usb' or 'network''

            dr-interface: The interface-type between the docking-station and data recorder.
						  This might be 'docked', 'optical' or 'wifi'
    """

    stream_stop = 0
    stream_start = 1
    
    def __init__(self, ds_interface, dr_interface, idx = -1):
        if (_tmsi_sdk == None):
            initialize()
        self._info = SagaInfo(ds_interface, dr_interface)
        self._config = SagaConfig()
        self._channels = [] # Active channel list
        self._imp_channels = [] # Impedance channel list
        self._id = _device_info_list[idx].id
        self._device_handle = DeviceHandle(0) #TMSiDeviceHandle
        self._idx_device_list_info = idx;
        self._last_error_code = TMSiDeviceRetVal.TMSI_OK
        self._sampling_thread = None
        self._measurement_type = MeasurementType.normal

    @property
    def id(self):
        """ 'int' : Unique id within all available devices. The id can be used to
            register as a client at the 'sample_data_server' for retrieval of
            sample-data of a specific device
        """
        return self._id

    @property
    def info(self):
        """ 'class DeviceInfo' : Static information of a device like used interfaces, serial numbers
        """
        dev_info = DeviceInfo(self._info.ds_interface, self._info.dr_interface)
        return dev_info


    @property
    def status(self):
        """ 'class DeviceStatus' : Runtime information of a device like device state
        """
        dev_status = DeviceStatus(self._info.state, self._last_error_code.value)
        return dev_status

    @property
    def channels(self):
        """ 'list of class DeviceChannel' : The list of enabled channels.
            Enabled channels are active during an 'normal' measurement.
        """
        chan_list = []
        for ch in self._config._channels:
            if (ch.enabled == True):
                sensor = ch.sensor
                if (ch.sensor != None):
                    sensor = DeviceSensor(ch.sensor.idx_total_channel_list,
                                          ch.sensor.id,
                                          ch.sensor.serial_nr,
                                          ch.sensor.product_id,
                                          ch.sensor.name,
                                          ch.sensor.unit_name,
                                          ch.sensor.exp)
                dev_ch = DeviceChannel(ch.type, ch.sample_rate, ch.alt_name, ch.unit_name, (ch.chan_divider != -1), ch.bandwidth, sensor)
                chan_list.append(dev_ch)
        return chan_list

    @property
    def imp_channels(self):
        """ 'list of class DeviceChannel' : The list of impedance channels.
            Impedance channels are active during an 'impedance' measurement.
        """
        imp_chan_list = []
        for ch in self._imp_channels:
            dev_ch = DeviceChannel(ch.type, 0, ch.alt_name, 'kOhm', True)
            imp_chan_list.append(dev_ch)
        return imp_chan_list
    @property

    def sensors(self):
        """ 'list of class DeviceSensor' : The complete list of sensor-information
            for the  sensor-type channels : BIP and AUX
        """
        sensor_list = []
        for sensor in self._sensor_list:
            dev_sensor = DeviceSensor(sensor.idx_total_channel_list,
                                      sensor.id,
                                      sensor.serial_nr,
                                      sensor.product_id,
                                      sensor.name,
                                      sensor.unit_name,
                                      sensor.exp)
            sensor_list.append(dev_sensor)
        return sensor_list

    @property
    def config(self):
        """ 'class DeviceConfigs' : The configuration of a device which exists
            out of individual properties (like base-sample-rate) and the total
            channel list (with enabled and disabled channels)
        """
        self._config._parent = self
        return self._config

    @property
    def datetime(self):
        """ 'datetime' Current date and time of the device
        """
        if (self._info.state == DeviceState.disconnected):
            raise TMSiError(TMSiErrorCode.device_not_connected)

        dev_full_status_report = TMSiDevFullStatReport()
        dev_bat_report = (TMSiDevBatReport * _MAX_NUM_BATTERIES)()
        dev_time = TMSiTime()
        dev_storage_report = TMSiDevStorageReport()

        self._last_error_code = _tmsi_sdk.TMSiGetFullDeviceStatus(self._device_handle,
                                                         pointer(dev_full_status_report),
                                                         pointer(dev_bat_report),
                                                         _MAX_NUM_BATTERIES,
                                                         pointer(dev_time),
                                                         pointer(dev_storage_report))
        if (self._last_error_code == TMSiDeviceRetVal.TMSI_OK):
            return datetime.datetime(dev_time.Year + 1900, dev_time.Month + 1, dev_time.DayOfMonth, dev_time.Hours, dev_time.Minutes, dev_time.Seconds)
        else:
            raise TMSiError(TMSiErrorCode.device_error)

    @datetime.setter
    def datetime(self, dt):
        """ 'datetime' Sets date and time of the device
        """
        if (self._info.state == DeviceState.disconnected):
            raise TMSiError(TMSiErrorCode.device_not_connected)

        dev_time = TMSiTime()
        dev_time.Year = dt.year - 1900
        dev_time.Month = dt.month - 1
        dev_time.DayOfMonth = dt.day
        dev_time.Hours = dt.hour
        dev_time.Minutes = dt.minute
        dev_time.Seconds = dt.second
        self._last_error_code = _tmsi_sdk.TMSiSetDeviceRTC(self._device_handle, pointer(dev_time))
        if (self._last_error_code != TMSiDeviceRetVal.TMSI_OK):
            raise TMSiError(TMSiErrorCode.device_error)


    def open(self):
        """ Opens the connection to the device.

            A connection will be established with the available system.

            The functionailty a device offers will only be available when a connection
            to the system has been established.
        """
        if (self._id != SagaConst.TMSI_DEVICE_ID_NONE):
            # A device is found. Open the connection and adapt the information of the opened device
            self._last_error_code = _tmsi_sdk.TMSiOpenDevice(pointer(self._device_handle), self._id, self.info.dr_interface.value)
            if (self._last_error_code == TMSiDeviceRetVal.TMSI_DS_DEVICE_ALREADY_OPEN):
                # The found device is available but in it's open-state: Close and re-open the connection
                self._last_error_code = _tmsi_sdk.TMSiCloseDevice(self._device_handle)
                self._last_error_code = _tmsi_sdk.TMSiOpenDevice(pointer(self._device_handle), self._id, self.info.dr_interface.value)

            if (self._last_error_code == TMSiDeviceRetVal.TMSI_OK):
                # The device is opened succesfully. Update the device information.
                _device_info_list[self._idx_device_list_info].state = DeviceState.connected

                self._info.state = DeviceState.connected
                self._info.id = self._id
                self._info.ds_interface = _device_info_list[self._idx_device_list_info].ds_interface
                self._info.dr_interface = _device_info_list[self._idx_device_list_info].dr_interface
                self._info.ds_serial_number = _device_info_list[self._idx_device_list_info].ds_serial_number
                self._info.dr_serial_number = _device_info_list[self._idx_device_list_info].dr_serial_number

                # Read the device's configuration
                self.__read_config_from_device()
                
                _recordings_list = self.get_device_storage_list()
                if _recordings_list:
                    warnings.warn("\n\n!!! \nThere is/are recordings stored on the onboard memory. Changing the configuration will clear the device's SD card!\nDo you want to continue? ('yes'/'y' continue, all others abort opening)\n!!!\n", stacklevel = 1)
                    abort = input('Continue?\n')
                    
                    if abort.lower() == 'yes' or abort.lower() == 'y':
                        pass
                    else:
                        raise TMSiError(TMSiErrorCode.general_error)
                
                if self.info.dr_interface == DeviceInterfaceType.wifi:
                    bandwidth = self.get_current_bandwidth()
                    if bandwidth > self.config._interface_bandwidth:
                        warnings.warn('\n\n!!! \nThe maximum Wi-Fi bandwidth is exceeded. Please change the device configuration by disabling channels or altering the sample rate(s).\n!!!\n', stacklevel = 1)

            else:
                raise TMSiError(TMSiErrorCode.device_error)
        else:
            raise TMSiError(TMSiErrorCode.no_devices_found)

    def close(self):
        """ Closes the connection to the device.
        """
        if (self._info.state != DeviceState.disconnected):

            self._last_error_code = _tmsi_sdk.TMSiCloseDevice(self._device_handle)
            _device_info_list[self._idx_device_list_info].state = DeviceState.disconnected
            self._info.state = DeviceState.disconnected
        else:
            raise TMSiError(TMSiErrorCode.device_not_connected)

    def start_measurement(self, measurement_type = MeasurementType.normal):
        """ Starts a measurement on the device.
            Clients, which want to receive the sample-data of a measurement,
            must be registered at the 'sample data server' before the measurement is started.

        Args:
            measurement_type : - MeasurementType.normal (default), starts a measurement
                                    with the 'enabled' channels: 'Device.channels[]'.
                               - MeasurementType.impedance, starts an impedance-measurement
                                    with all 'impedance' channels: 'Device.imp_channels[]'
        """
        # Only allow one measurement at the same time
        if (self._info.state == DeviceState.sampling):
            raise TMSiError(TMSiErrorCode.api_invalid_command)
        if (self._info.state != DeviceState.connected):
            raise TMSiError(TMSiErrorCode.device_not_connected)

        if self.get_current_bandwidth() > self.config.interface_bandwidth:
            print('\n\nA measurement could not be started.\nThe maximum Wi-Fi bandwidth is exceeded. Please change the device configuration by disabling channels or altering the sample rate(s).\n\n')
            raise TMSiError(TMSiErrorCode.api_invalid_command)
            
        self._measurement_type = measurement_type
        _tmsi_sdk.TMSiResetDeviceDataBuffer(self._device_handle)

        # Create and start the sampling-thread to capture and process incoming measurement-data,
        # For a normal measurement sample_conversion must be applied
        self._sampling_thread = _SamplingThread(name='producer-' + str(self._device_handle.value))
        if (self._measurement_type == MeasurementType.normal):
            self._sampling_thread.initialize(self._device_handle, self._channels, True)
        else:
            self._sampling_thread.initialize(self._device_handle, self._imp_channels, False)

        self._conversion_thread = _ConversionThread(sampling_thread = self._sampling_thread)

        self._sampling_thread.start()
        self._conversion_thread.start()

        # Request to start the measurement at the device
        if (self._measurement_type == MeasurementType.normal):
            measurement_request = TMSiDevSampleReq()
            measurement_request.SetSamplingMode = 1
            measurement_request.DisableAvrRefCalc = 0
            self._last_error_code = _tmsi_sdk.TMSiSetDeviceSampling(self._device_handle, pointer(measurement_request) )
        else:
            measurement_request = TMSiDevImpReq()
            measurement_request.SetImpedanceMode = 1
            self._last_error_code = _tmsi_sdk.TMSiSetDeviceImpedance(self._device_handle, pointer(measurement_request) )

        if (self._last_error_code == TMSiDeviceRetVal.TMSI_OK):
            self._info.state = DeviceState.sampling
        else :
            # Measurement could not be started. Stop the sampling-thread and report error.
            self._sampling_thread.stop()
            raise TMSiError(TMSiErrorCode.device_error)

    def stop_measurement(self):
        """ Stops an ongoing measurement on the device."""
        if (self._info.state != DeviceState.sampling):
            raise TMSiError(TMSiErrorCode.api_invalid_command)

        if (self._measurement_type == MeasurementType.normal):
            measurement_request = TMSiDevSampleReq()
            measurement_request.SetSamplingMode = 0
            measurement_request.DisableAutoswitch = 0
            measurement_request.DisableRepairLogging = 0
            measurement_request.DisableAvrRefCalc = 0
            self._last_error_code = _tmsi_sdk.TMSiSetDeviceSampling(self._device_handle, pointer(measurement_request) )
        else:
            measurement_request = TMSiDevImpReq()
            measurement_request.SetImpedanceMode = 0
            self._last_error_code = _tmsi_sdk.TMSiSetDeviceImpedance(self._device_handle, pointer(measurement_request) )

        self._sampling_thread.stop()
        self._conversion_thread.stop()

        self._info.state = DeviceState.connected

    def set_factory_defaults(self):
        """ Initiates a factory reset to restore the systems' default configuration."""

        if (self._info.state == DeviceState.disconnected):
            raise TMSiError(TMSiErrorCode.device_not_connected)

        dev_set_config = TMSiDevSetConfig()

        dev_set_config.DRSerialNumber = 0
        dev_set_config.NrOfChannels = 0
        dev_set_config.SetBaseSampleRateHz = 0
        dev_set_config.SetConfiguredInterface = 0
        dev_set_config.SetTriggers = 0
        dev_set_config.SetRefMethod = 0
        dev_set_config.SetAutoRefMethod = 0
        dev_set_config.SetDRSyncOutDiv = 0
        dev_set_config.DRSyncOutDutyCycl = 0
        dev_set_config.SetRepairLogging = 0
        dev_set_config.StoreAsDefault = 0
        dev_set_config.WebIfCtrl = 0

        default_pin = bytearray('0000', 'utf-8')
        dev_set_config.PinKey[:] = default_pin[:]

        dev_set_config.PerformFactoryReset = 1

        dev_set_channel = TMSiDevSetChCfg()
        dev_set_channel.ChanNr = 0;
        dev_set_channel.ChanDivider = -1;
        # dev_set_channel.AltChanName[0] = '\0';

        self._last_error_code = _tmsi_sdk.TMSiSetDeviceConfig(self._device_handle, pointer(dev_set_config), pointer(dev_set_channel), 1);

        # Align the internal administration with the new device's configuration
        if (self._last_error_code == TMSiDeviceRetVal.TMSI_OK):
            self.__read_config_from_device()
        else:
            raise TMSiError(TMSiErrorCode.device_error)
            
        message="Please repower the Data Recorder to activate the factory settings. \n\n"\
            "To do this: \n\n"\
            "1. Undock the Data Recorder from the Docking Station. \n"\
            "2. Remove the batteries from the Data Recorder.\n"\
            "3. Wait for 5 seconds. \n"\
            "4. Insert the batteries again. \n"\
            "5. Dock the Data Recorder onto the Docking Station. \n"\
            "6. Press the Power-button of the Data Recorder untill the LED-indicators start flashing.\n"\
            "7. The default settings are now activated."
            
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning("Factory Reset", message)

    def save_config(self, filename):
        """ Saves the current device configuration to file.

            Args:
                filename : path and filename of the file to which the current
                           device configuration must be saved.
        """
        result = xml_write_config(filename, self._config)
        if (result != True):
            raise TMSiError(TMSiErrorCode.general_error)

    def load_config(self, filename):
        """ Loads a device configuration from file into the attached system.

            1. The device configuration is read from the specified file.
            2. This configuration is uploaded into the attached system.
            3. The configuration is downloaded from the system to be sure that
                the configuration of the Python-interface is in sync with the
                configuration of the device.

            Note : It is advised to "refresh" the applications' local "variables"
                   after a new device configuration has been load.

            Args:
                filename : path and filename of the file that must be loaded
        """
        if (self._info.state == DeviceState.disconnected):
            raise TMSiError(TMSiErrorCode.device_not_connected)

        # The load-config exists out of the next steps:
        #  1. Read the mutable device-configuration-settings from the xml-file
        #  2. Write these settings to the Saga device
        #  3. Read back the device configuration from the Saga-device. This to be sure
        #     that the configuration of the Saga-device and the Python-interface are in sync.
        result, read_xml_config = xml_read_config(filename)
        if (result == True):
            # The configuration has been successfully read, Now merge these configuration settings
            # with the unmutable configuration settings of the device
            # Do not overwrite configured interface
            read_xml_config._configured_interface=self._config._configured_interface
            self._config = read_xml_config
            self._update_config()
        else:
            print('Can not load .xml config file')
            raise TMSiError(TMSiErrorCode.general_error)

    def update_sensors(self):
        """ Called when sensors have been attached or detached to/from the device.
            The complete configuration including the new sensor-configuration
            is reloaded from the device.

        Note:
            It is advised to "refresh" the applications' local "variables"
            after the the complete configuration has been reloaded.
        """
        self.__read_config_from_device()

    def download_recording_file(self, file_id, verbosity = True):
        """download the file requested from the device

        :param file_id: ID of the file wanted to be downloaded
        :type file_id: string
        :param verbosity: print the download percentage, defaults to True
        :type verbosity: bool, optional
        """
        self.start_download_recording_file(file_id)

        while True:
            download_percentage = self.monitor_download_recording_file()
            if verbosity:
                print('\rProgress: % 0.1f %%' %(download_percentage), end="\r")
            if download_percentage >= 100:
                break
            time.sleep(0.1)
        
        self.stop_download_recording_file(file_id)

    def start_download_recording_file(self, file_id):
        """ Starts the download stream from the device of the file requested

        Args:
            file_id : ID of the file wanted to be downloaded
        """
        # Only allow one measurement at the same time
        if (self._info.state == DeviceState.sampling):
            raise TMSiError(TMSiErrorCode.api_invalid_command)
        if (self._info.state != DeviceState.connected):
            raise TMSiError(TMSiErrorCode.device_not_connected)

        _tmsi_sdk.TMSiResetDeviceDataBuffer(self._device_handle)

        # Create and start the sampling-thread to capture and process incoming measurement-data,
        # For a normal measurement sample_conversion must be applied
        self._sampling_thread = _SamplingThread(name='producer-' + str(self._device_handle.value))
        self._sampling_thread.initialize(self._device_handle, self._channels, True)
            
        self._conversion_thread = _ConversionThread(sampling_thread = self._sampling_thread)

        # Request to stop the downstream so that we can read the details of the recording
        recording_metadata = TMSiDevRecDetails()
        impedance_report_list = (TMSiDevImpReport*100)()
        impedance_report_list_len = 100

        self._last_error_code = _tmsi_sdk.TMSiGetRecordingFile(
            self._device_handle,
            file_id,
            SagaDevice.stream_stop,
            pointer(recording_metadata),
            pointer(impedance_report_list),
            impedance_report_list_len)

        self._sampling_thread.download_samples_limit = recording_metadata.NoOfSamples
        self._sampling_thread.downloaded_samples = 0
        self._sampling_thread.download_percentage = 0

        self._last_error_code = _tmsi_sdk.TMSiGetRecordingFile(
            self._device_handle,
            file_id,
            SagaDevice.stream_start,
            pointer(recording_metadata),
            pointer(impedance_report_list),
            impedance_report_list_len)

        if (self._last_error_code == TMSiDeviceRetVal.TMSI_OK.value):
            self._info.state = DeviceState.sampling
        else :
            # Measurement could not be started. Stop the sampling-thread and report error.
            self._sampling_thread.stop()
            raise TMSiError(TMSiErrorCode.device_error)

        self._sampling_thread.start()
        self._conversion_thread.start()
        
    def monitor_download_recording_file(self):
        return self._sampling_thread.download_percentage

    def stop_download_recording_file(self, file_id):
        """ Stops the download stream from the device of the file requested

        Args:
            file_id : ID of the file wanted to be downloaded
        """
        if (self._info.state != DeviceState.sampling):
            raise TMSiError(TMSiErrorCode.api_invalid_command)

        recording_metadata = TMSiDevRecDetails()
        impedance_report_list = (TMSiDevImpReport*100)()
        impedance_report_list_len = 100

        self._last_error_code = _tmsi_sdk.TMSiGetRecordingFile(
            self._device_handle,
            file_id,
            SagaDevice.stream_stop,
            pointer(recording_metadata),
            pointer(impedance_report_list),
            impedance_report_list_len)

        self._sampling_thread.stop()
        self._conversion_thread.stop()

        self._info.state = DeviceState.connected
    
    def get_device_memory_configuration(self):
        return self._get_device_amb_config()

    def set_device_recording_button(self, prefix_name=None):
        """enable recording button to save on internal memory

        :param prefix_name: prefix name for the files, defaults to None
        :type prefix_name: string, optional
        """
        card_config_interface_bandwidth = 2 # MBit/s
        # Check whether the bandwidth does not exceed 2 Mbit/s
        if self.get_current_bandwidth() > card_config_interface_bandwidth:
            print('\n\nButton start could not be enabled.\nThe maximum bandwidth (2Mbit/s) is exceeded. Please change the device configuration by disabling channels or altering the sample rate(s).\n\n')
            raise TMSiError(TMSiErrorCode.api_invalid_command)
        
        device_amb_conf = self.get_device_memory_configuration()
        device_amb_conf.StartControl = 8
        device_amb_conf.EndControl = 0
        if prefix_name is not None:
            max_len = len(prefix_name)
        if max_len>16:
            max_len = 16
        prefix_name = bytearray(prefix_name, 'utf-8')
        name = bytearray(16)
        name[:max_len] = prefix_name[:max_len]
        device_amb_conf.PrefixFileName[:] = name
        self._last_error_code = self._set_device_amb_config(device_amb_conf)
    
    def set_device_backup_logging(self, prefix_name=None):
        """enable backup logging to save on internal memory every time a recording is performed

        :param prefix_name: prefix name for the files, defaults to None
        :type prefix_name: string, optional
        """
        device_amb_conf = self.get_device_memory_configuration()
        device_amb_conf.StartControl = 16
        device_amb_conf.EndControl = 0
        if prefix_name is not None:
            max_len = len(prefix_name)
        if max_len>16:
            max_len = 16
        prefix_name = bytearray(prefix_name, 'utf-8')
        name = bytearray(16)
        name[:max_len] = prefix_name[:max_len]
        device_amb_conf.PrefixFileName[:] = name
        self._last_error_code = self._set_device_amb_config(device_amb_conf)

    def set_device_backup_disabled(self):
        """disable backup logging"""
        device_amb_conf = self.get_device_memory_configuration()
        device_amb_conf.StartControl = 0
        self._last_error_code = self._set_device_amb_config(device_amb_conf)
    
    def _get_device_amb_config(self):
        """
        Gets the configuration of the flash memory on the SAGA
        
        :raise TMSiErrorCode.device_error: If the get configuration fails.
        :return: The configuration structure.
        :rtype: TMSiDevRecCfg
        
        """
        device_amb_cfg = TMSiDevRecCfg()
        self._last_error_code = _tmsi_sdk.TMSiGetDeviceAmbConfig(
            self._device_handle,
            pointer(device_amb_cfg)
        )
        if (self._last_error_code == TMSiDeviceRetVal.TMSI_OK.value):
            return device_amb_cfg
        else:
            raise TMSiError(TMSiErrorCode.device_error)
    
    def _set_device_amb_config(self, cfg):
        """
        Sets the configuration of the flash memory on the SAGA
        
        :param cfg: configuration to send to the SAGA.
        :type cfg: TMSiDevRecCfg
        :raise TMSiErrorCode.device_error: If the set configuration fails.
        :return: TMSiDeviceRetVal.TMSI_OK
        :rtype: TMSiDeviceRetVal
        
        """
        self._last_error_code = _tmsi_sdk.TMSiSetDeviceAmbConfig(
            self._device_handle,
            pointer(cfg)
        )
        if (self._last_error_code == TMSiDeviceRetVal.TMSI_OK.value):
            return TMSiDeviceRetVal.TMSI_OK
        else:
            raise TMSiError(TMSiErrorCode.device_error)

    def get_device_storage_list(self):
        """
        Returns the list of files currently available on the SAGA device
        
        :raise TMSiErrorCode.device_error: If the get storage list fails.
        :return: A list of available recording present on the Data Recorder.
        :rtype: TMSiDevRecList
        
        """
        device_config = TMSiDevGetConfig()
        device_channel_list = (TMSiDevChDesc * self._config.num_channels)()
        self._last_error_code = _tmsi_sdk.TMSiGetDeviceConfig(
            self._device_handle, 
            pointer(device_config), 
            pointer(device_channel_list), 
            self._config.num_channels)
        if (self._last_error_code != TMSiDeviceRetVal.TMSI_OK):
            raise TMSiError(TMSiErrorCode.device_error)

        available_recs = device_config.AvailableRecordings
        recordings_list = (TMSiDevRecList*available_recs)()
        recordings_len = c_ulong()
        recordings_dict = {}
        self._last_error_code = _tmsi_sdk.TMSiGetDeviceStorageList(
            self._device_handle, 
            pointer(recordings_list), available_recs, pointer(recordings_len))
        if (self._last_error_code == TMSiDeviceRetVal.TMSI_OK):
            for recording in recordings_list:
                recordings_dict[recording.RecFileID] = recording.RecFileName
            return recordings_dict
        else:
            raise TMSiError(TMSiErrorCode.device_error)
            
            
    def get_current_bandwidth(self):
        """
        Returns the bandwidth (in MBits/sec) in use with the current device 
        configuration
        """
        
        ch_bandwidth = 0
        ch_bandwidth = sum([ch_bandwidth + ch.bandwidth for ch in self.channels])
        
        # Get the used bandwidth in 
        used_bandwidth = (80 * self.config.get_sample_rate(ChannelType.all_types) + 
                          ch_bandwidth) / 1e6
        self._used_bandwidth = used_bandwidth
        
        return used_bandwidth
    
    def _update_config(self):
        # Upload the configuration to the device and always download it again
        # to certify that sdk-configuration and device-configuration keep in sync
        self.__write_config_to_device()
        self.__read_config_from_device()

    def __read_config_from_device(self):
        # Reset device configuration
        self._channels = []
        self._imp_channels = []
        self._config._channels = []
        for idx in range(0, len(self._config._sample_rates)):
            self._config._sample_rates[idx].sample_rate = 0

        # Retrieve configuration settings and the channel list
        device_status_report = TMSiDevStatReport()
        self._last_error_code = _tmsi_sdk.TMSiGetDeviceStatus(self._device_handle, pointer(device_status_report))
        if (self._last_error_code == TMSiDeviceRetVal.TMSI_OK):
            self._config._num_channels = device_status_report.NrOfChannels

            device_config = TMSiDevGetConfig()
            device_channel_list = (TMSiDevChDesc * self._config.num_channels)()
            self._last_error_code = _tmsi_sdk.TMSiGetDeviceConfig(self._device_handle, pointer(device_config), pointer(device_channel_list), self._config.num_channels)
            if (self._last_error_code == TMSiDeviceRetVal.TMSI_OK):
                self._config._base_sample_rate = device_config.BaseSampleRateHz
                self._config._configured_interface = device_config.ConfiguredInterface
                self._config._triggers = device_config.TriggersEnabled
                self._config._reference_method = device_config.RefMethod
                self._config._auto_reference_method = device_config.AutoRefMethod
                self._config._dr_sync_out_divider = device_config.DRSyncOutDiv
                self._config._dr_sync_out_duty_cycle = device_config.DRSyncOutDutyCycl
                self._config._repair_logging = device_config.RepairLogging
                self._config._num_sensors = device_config.NrOfSensors
                self._config._interface_bandwidth = device_config.InterFaceBandWidth

                for i in range(self._config.num_channels):
                    channel = SagaChannel()
                    channel.type = ChannelType(device_channel_list[i].ChannelType)
                    channel.format = device_channel_list[i].ChannelFormat
                    channel.chan_divider = device_channel_list[i].ChanDivider
                    channel.enabled = (channel.chan_divider != -1)
                    channel.sample_rate = device_config.BaseSampleRateHz
                    if (device_channel_list[i].ChanDivider != -1):
                        for j in range(channel.chan_divider):
                            channel.sample_rate = int(channel.sample_rate / 2)
                        if (self._config._sample_rates[channel.type.value].sample_rate == 0):
                            self._config._sample_rates[channel.type.value].sample_rate = channel.sample_rate
                            self._config._sample_rates[channel.type.value].chan_divider = channel.chan_divider
                    channel.imp_divider = device_channel_list[i].ImpDivider
                    channel.bandwidth = device_channel_list[i].ChannelBandWidth
                    channel.exp = device_channel_list[i].Exp
                    channel.unit_name = device_channel_list[i].UnitName.decode('windows-1252')
                    channel.def_name = device_channel_list[i].DefChanName.decode('windows-1252')
                    channel.alt_name = device_channel_list[i].AltChanName.decode('windows-1252')
                    self._config._channels.append(channel)
                    if channel.chan_divider != -1:
                        # Active channel
                        self._channels.append(channel)
                    if channel.imp_divider != -1:
                        # Impedance channel
                        self._imp_channels.append(channel)

                # The systems' sample rate is equal to the sample rate of the COUNTER-channel,
                # which is always the last channel in the active channel list
                self._config._sample_rate = self._channels[len(self._channels) - 1].sample_rate

                # Read the sensor data, parse the sensor-metadata and when appropriate
                # attach a SagaSensor-object to the SagaChannel.
                device_sensor_list = (TMSiDevGetSens * self._config._num_sensors)()
                sensor_list_len = c_ulong()
                self._last_error_code = _tmsi_sdk.TMSiGetDeviceSensor(self._device_handle, pointer(device_sensor_list), self._config._num_sensors, pointer(sensor_list_len))
                if (self._last_error_code == TMSiDeviceRetVal.TMSI_OK):
                    self._update_sensor_info(device_sensor_list, sensor_list_len)
                else:
                    # Failure TMSiGetDeviceSensor()
                    raise TMSiError(TMSiErrorCode.device_error)

            else:
                # Failure TMSiGetDeviceConfig()
                raise TMSiError(TMSiErrorCode.device_error)

        else:
            # Failure TMSiGetDeviceStatus()
            raise TMSiError(TMSiErrorCode.device_error)

    def __write_config_to_device(self):
        # Upload the current sdk-configuration to the device, mark it also
        # as the new default configuration
        dev_set_config = TMSiDevSetConfig()

        dev_set_config.DRSerialNumber = self._info.dr_serial_number
        dev_set_config.NrOfChannels = self._config._num_channels
        dev_set_config.SetBaseSampleRateHz = self._config._base_sample_rate
        dev_set_config.SetConfiguredInterface = self._config._configured_interface
        dev_set_config.SetTriggers = self._config._triggers
        dev_set_config.SetRefMethod = self._config._reference_method
        dev_set_config.SetAutoRefMethod = self._config._auto_reference_method
        dev_set_config.SetDRSyncOutDiv = self._config._dr_sync_out_divider
        dev_set_config.DRSyncOutDutyCycl = self._config._dr_sync_out_duty_cycle
        dev_set_config.SetRepairLogging = self._config._repair_logging
        dev_set_config.StoreAsDefault = 1 # Store always as default configuration
        dev_set_config.WebIfCtrl = 0

        default_pin = bytearray('0000', 'utf-8')
        dev_set_config.PinKey[:] = default_pin[:]

        dev_set_config.PerformFactoryReset = 0

        dev_channel_list = (TMSiDevSetChCfg * self._config._num_channels)()
        for idx, saga_channel in enumerate(self._config._channels):

            dev_channel_list[idx].ChanNr = idx
            dev_channel_list[idx].ChanDivider = saga_channel.chan_divider
            max_len = len( saga_channel.alt_name)
            name = bytearray(saga_channel.alt_name, 'utf-8')
            dev_channel_list[idx].AltChanName[:max_len] = name[:max_len]

        self._last_error_code = _tmsi_sdk.TMSiSetDeviceConfig(self._device_handle, pointer(dev_set_config), pointer(dev_channel_list), self._config.num_channels);
        if (self._last_error_code != TMSiDeviceRetVal.TMSI_OK):
            # Failure TMSiSetDeviceConfig()
            raise TMSiError(TMSiErrorCode.device_error)

    def _update_sensor_info(self, device_sensor_list, sensor_list_len):
        # 1. Update the actual sensor list

        # Reset the current list
        self._sensor_list = []
        for i in range(sensor_list_len.value):
            sensor = SagaSensor()

            sensor.idx_total_channel_list = device_sensor_list[i].ChanNr
            sensor.id = device_sensor_list[i].SensorID
            sensor.IOMode = device_sensor_list[i].IOMode

            # 2. Parse the sensor metadata
            idx = 0
            manufacturer_id, serial_nr, product_id, channel_count, additional_structs = struct.unpack_from('<HIQBB', device_sensor_list[i].SensorMetaData, idx)
            if (channel_count > 0):
                if (self._config._channels[sensor.idx_total_channel_list].type == ChannelType.AUX):
                    # It concerns an AUX-channel-group: AUX-1, AUX-2 or AUX-3
                    sensor.manufacturer_id = manufacturer_id
                    sensor.serial_nr = serial_nr
                    sensor.product_id = product_id
                    idx += 16
                    for j in range(channel_count):
                        struct_id = struct.unpack_from('<H', device_sensor_list[i].SensorMetaData, idx)
                        # Parse the data if it concerns a 'SensorDefaultChannel'
                        if (struct_id[0] == 0x0000):
                            idx += 2
                            chan_name, unit_name, exp, gain, offset = struct.unpack_from('<10s10shff', device_sensor_list[i].SensorMetaData, idx)

                            sensor.name = chan_name
                            sensor.unit_name = unit_name
                            sensor.exp = exp
                            sensor.gain = gain
                            sensor.offset = offset

                            # append sensor-into to the device-sensor-list and ...
                            self._sensor_list.append(copy(sensor))
                            # attach a copy of the sensor-object also to the specified channel
                            self._config._channels[sensor.idx_total_channel_list].sensor = copy(sensor)
                            # Overrule the channel's alt_name with the name of the sensor-channel
                            self._config._channels[sensor.idx_total_channel_list].alt_name = sensor.name
                            # Overrule the channel's unit_name with the uint name of the sensor-channel
                            self._config._channels[sensor.idx_total_channel_list].unit_name = sensor.unit_name

                            # Prepare for next sensor-channel
                            sensor.idx_total_channel_list += 1
                            idx += 30
                        else:
                            # ran into an 'empty struct', no more sensor info expected
                            break
            else:
                # Always add the sensor-data to the device-sensor-list
                self._sensor_list.append(copy(sensor))
                # Add sensor-object to both BIP-channels when a sensor is detected on a BIP-channel
                if (self._config._channels[sensor.idx_total_channel_list].type == ChannelType.BIP) and (sensor.id != -1):
                    self._config._channels[sensor.idx_total_channel_list].sensor = copy(sensor)
                    sensor.idx_total_channel_list += 1
                    self._sensor_list.append(copy(sensor))
                    self._config._channels[sensor.idx_total_channel_list+1].sensor = copy(sensor)

def initialize():
    """Initialize the interface-environment."""

    # - Load the device-interface-library
    # - Create local list of 2 devices.
    try:
        global _tmsi_sdk
        global _device_info_list
        _tmsi_sdk = SagaSDK
        print(_tmsi_sdk)
        for i in range (_MAX_NUM_DEVICES):
            _device_info_list.append(SagaInfo())
    except:
        _tmsi_sdk = None
        raise TMSiError(TMSiErrorCode.api_no_driver)

def _discover(ds_interface, dr_interface):
        # 1. Executes a discovery on devices based on the given interfaces for DS and DR
        # 2. Updates the local device-list with the result
        device_list = (TMSiDevList * _MAX_NUM_DEVICES)()
        
        if dr_interface == DeviceInterfaceType.wifi:
            _num_retries = 10
        else:
            _num_retries = 5
        for i in range (_MAX_NUM_DEVICES):
            device_list[i].TMSiDeviceID = SagaConst.TMSI_DEVICE_ID_NONE
        
        while _num_retries > 0:
            ret = _tmsi_sdk.TMSiGetDeviceList(pointer(device_list), _MAX_NUM_DEVICES, ds_interface.value, dr_interface.value )

            if (ret == TMSiDeviceRetVal.TMSI_OK):
                # Devices are found, update the local device list with the found result
                for i in range (_MAX_NUM_DEVICES):
                    if (device_list[i].TMSiDeviceID != SagaConst.TMSI_DEVICE_ID_NONE):
                        for ii in range (_MAX_NUM_DEVICES):
                            if (_device_info_list[ii].id == SagaConst.TMSI_DEVICE_ID_NONE):
                                _device_info_list[ii].id = device_list[i].TMSiDeviceID
                                _device_info_list[ii].ds_interface = ds_interface
                                _device_info_list[ii].dr_interface = dr_interface
                                _device_info_list[ii].ds_serial_number = device_list[i].DSSerialNr
                                _device_info_list[ii].dr_serial_number = device_list[i].DRSerialNr
                                _device_info_list[ii].state = DeviceState.disconnected
                                
                                _num_retries = 0
                                break
                _num_retries -= 1
            else: 
                _num_retries -= 1
                print('Trying to open connection to device. Number of retries left: ' + str(_num_retries))
                time.sleep(0.5)
                if _num_retries == 0:
                    raise TMSiError(TMSiErrorCode.no_devices_found)


def discover(ds_interface, dr_interface):
    if not SagaDllAvailable:
        print("SAGA DLL not available.")
        raise TMSiError(code=TMSiErrorCode.missing_dll)
    if SagaDllLocked:
        print("SAGA DLL already in use.")
        raise TMSiError(code=TMSiErrorCode.already_in_use_dll)
    discoveryList = []

    if (_tmsi_sdk == None):
        initialize()

    _discover(ds_interface, dr_interface)
    for idx in range (_MAX_NUM_DEVICES):
        if (_device_info_list[idx].id != SagaConst.TMSI_DEVICE_ID_NONE):
            discoveryList.append(SagaDevice(ds_interface, dr_interface, idx))

    return discoveryList


class _SamplingThread(threading.Thread):
    def __init__(self, name):
        super(_SamplingThread,self).__init__()
        self.name = name
        self._device_handle = None
        self.sample_data_buffer_size = 409600
        self.sample_data_buffer = (c_float * self.sample_data_buffer_size)(0)
        self.retrieved_sample_sets = (c_uint)(0)
        self.retrieved_data_type = (c_int)(0)
        self.num_samples_per_set = 0
        self.channels = []
        self.sample_conversion = False
        
        
    def run(self):
        print(self.name," started")
        self.sampling = True;
        
        while self.sampling:
            ret = _tmsi_sdk.TMSiGetDeviceData(self._device_handle, pointer(self.sample_data_buffer), self.sample_data_buffer_size, pointer(self.retrieved_sample_sets), pointer(self.retrieved_data_type) )
            if (ret == TMSiDeviceRetVal.TMSI_OK):
                if self.retrieved_sample_sets.value > 0:
                    self.conversion_queue.put((deepcopy(self.sample_data_buffer), self.retrieved_sample_sets.value))
                    
            if hasattr(self, "download_samples_limit"):
                self.downloaded_samples += self.retrieved_sample_sets.value
                # print("added {} downloaded {} expected {}".format(
                #         self.retrieved_sample_sets.value, 
                #         self.downloaded_samples,
                #         self.download_samples_limit))
                self.download_percentage = self.downloaded_samples * 100.0 / self.download_samples_limit
                if self.downloaded_samples >= self.download_samples_limit:
                    break
            time.sleep(0.050)
            
        print(self.name, " ready")
        

    def initialize(self, device_handle, channels, sample_conversion):
        self._device_handle = device_handle
        self.channels = channels
        self.num_samples_per_set = len(channels)
        self.sample_conversion = sample_conversion

        _MAX_SIZE_CONVERSION_QUEUE = 50
        self.conversion_queue = queue.Queue(_MAX_SIZE_CONVERSION_QUEUE)        

    def stop(self):
        print(self.name, " stop sampling")
        self.sampling = False;
        
        
class _ConversionThread(threading.Thread):
    def __init__(self, sampling_thread):
        super(_ConversionThread, self).__init__()
        self.channels = sampling_thread.channels
        self._device_handle = sampling_thread._device_handle
        self.q = sampling_thread.conversion_queue
        
        self.num_samples_per_set = sampling_thread.num_samples_per_set
        self.sample_conversion = sampling_thread.sample_conversion
        if self.sample_conversion:
            self._warn_message = True
        else:
            self._warn_message = False
            
        # sort channels per channel type
        self._float_chan=[]   
        self._sensor_chan=[]
        self._basic_conversion={}
        for j in range(len(self.channels)):
            if (self.channels[j].format == 0x0020):
                self._float_chan.append(j)
            else:
                if (self.sample_conversion):
                    if (self.channels[j].type == ChannelType.AUX) and (self.channels[j].sensor != None):
                        self._sensor_chan.append(j)
                    else:
                        conversion_factor=10**self.channels[j].exp
                        if not conversion_factor in self._basic_conversion:
                            self._basic_conversion[conversion_factor]=[j]
                        else:
                            self._basic_conversion[conversion_factor].append(j)
        
    def run(self):
        self.sampling = True
        
        if self._warn_message:
            # Initialise a sample counter that keeps track of whether samples might be lost
            last_counter = 0
        
        while (self.sampling) or (not self.q.empty()):
            while (not self.q.empty()):
                
                sample_data_buffer, retrieved_sample_sets = self.q.get()
                
                
                if (retrieved_sample_sets > 0):
                    # reshape data to matrix
                    sample_mat=np.reshape(sample_data_buffer[:self.num_samples_per_set*retrieved_sample_sets], (self.num_samples_per_set, retrieved_sample_sets), order='F')
                    if (self.sample_conversion):
                        #basic unit conversion
                        for conversion_factor in self._basic_conversion.keys():
                            sample_mat[self._basic_conversion[conversion_factor]]=sample_mat[self._basic_conversion[conversion_factor]]/conversion_factor
                        #sensor data conversion
                        for j in self._sensor_chan:
                            x= sample_mat[j]
                            sensor = self.channels[j].sensor
                            x = ((x + sensor.offset) * sensor.gain)/ (10**sensor.exp)
                            sample_mat[j]=x
                    
                    #conversion of float channels
                    for j in self._float_chan:
                        sample_mat[j]=float_to_uint(sample_mat[j]) 
                    
                    # Check sample counter integrity
                    if self._warn_message:
                        counterstep=np.diff(sample_mat[-1])
                        if not np.all(counterstep==1)or not ((sample_mat[-1,0]-last_counter)==1):
                            warnings.warn('\n\n!!! \nSomething is wrong, samples might be lost..\n!!!\n', stacklevel = 1)
                            self._warn_message = False
                        last_counter=copy(sample_mat[-1,-1])
                    
                    samples=sample_mat.flatten('F')
                    samples=samples.tolist()
                    
                    sd = sample_data.SampleData(retrieved_sample_sets, self.num_samples_per_set, samples )
                    sample_data_server.putSampleData(self._device_handle.value, sd)

            time.sleep(0.010)
        
    def stop(self):
        self.sampling = False
        

def float_to_uint(f):
    fmt_pack='f'*len(f)
    fmt_unpack='I'*len(f)
    pack_struct=struct.Struct(fmt_pack)
    return struct.unpack(fmt_unpack, pack_struct.pack(*list(f)))




