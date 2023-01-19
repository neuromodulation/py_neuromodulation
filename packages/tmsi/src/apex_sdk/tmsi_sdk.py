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
 * @file tmsi_sdk.py 
 * @brief 
 * Singleton class which handles the discovery of TMSi devices.
 */


'''

from .tmsi_utilities.singleton import Singleton
from .device.tmsi_device_enums import *
from .device.devices.apex.apex_device import ApexDevice


class TMSiSDK(metaclass = Singleton):
    """Singleton class which handles the discovery of TMSi devices"""
    def __init__(self):
        """Initializes the object."""
        self.__apex_device_list = []
        self.__apex_dongle_list = []
        self.__saga_device_list = []
        
    def discover(self, 
        dev_type = DeviceType.apex, 
        dr_interface = DeviceInterfaceType.usb, 
        ds_interface = DeviceInterfaceType.none,
        num_retries = 3):
        """Discovers if there are available devices.

        :param dev_type: device tipe to search, defaults to DeviceType.apex
        :type dev_type: DeviceType, optional
        :param dr_interface: datarecorder interface, defaults to DeviceInterfaceType.usb
        :type dr_interface: DeviceInterfaceType, optional
        :param ds_interface: dockstation interface (if needed), defaults to DeviceInterfaceType.none
        :type ds_interface: DeviceInterfaceType, optional
        :param num_retries: number of retry if nothing found
        :type num_retries: int, optional
        :return: _description_
        :rtype: tuple[list[TMSiDevice], list[TMSiDongle]]
        """
        if dev_type == DeviceType.apex:
            ApexDevice.discover(dr_interface, num_retries)
            self.__apex_device_list = ApexDevice.get_device_list(dr_interface)
            self.__apex_dongle_list = ApexDevice.get_dongle_list()
            return (self.__apex_device_list, self.__apex_dongle_list)
    
    def get_device_list(self, dev_type = DeviceType.apex):
        """Gets the list of available devices.

        :param dev_type: device to get, defaults to DeviceType.apex
        :type dev_type: DeviceType, optional
        :return: list of available devices.
        :rtype: list[TMSiDevice]
        """
        if dev_type == DeviceType.apex:
            return self.__apex_device_list
        elif dev_type == DeviceType.saga:
            return self.__saga_device_list

    def get_dongle_list(self, dev_type = DeviceType.apex):
        """Gets the list of available dongles.

        :param dev_type: device type dongle to get, defaults to DeviceType.apex
        :type dev_type: DeviceType, optional
        :return: list of available dongles.
        :rtype: list[TMSiDongle]
        """
        if dev_type == DeviceType.apex:
            return self.__apex_dongle_list

    def get_driver_version(self, dev_type = DeviceType.apex) -> str:
        """Gets the driver version

        :param dev_type: the device type for the drivers, defaults to DeviceType.apex
        :type dev_type: DeviceType, optional
        :return: driver version
        :rtype: str
        """
        if dev_type == DeviceType.apex:
            version = ApexDevice.get_driver_version()
            dll_version = "".join([chr(i) for i in version.DllVersionString if i != 0])
            usb_version = "".join([chr(i) for i in version.LibUsbVersionString if i != 0])
            return (dll_version, usb_version)
        
            
