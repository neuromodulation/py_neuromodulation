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
 * @file ${tmsi_device.py} 
 * @brief Starting module to instantiate <Device>-objects of TMSi measurement 
 * systems.
 *
 */


'''

from enum import Enum

from TMSiSDK.error import TMSiError, TMSiErrorCode
from TMSiSDK.device import DeviceInterfaceType
from TMSiSDK.devices.saga import saga_device
from TMSiSDK.devices.saga.saga_device import SagaDevice

from TMSiSDK import settings

class DeviceType(Enum):
    none = 0
    saga = 1

def initialize():
    """Initializes the TMSi-SDK environment.
        This must be done once before starting using the SDK.
    """
    settings._initialize()

def create(dev_type, dr_interface, ds_interface = DeviceInterfaceType.none):
    """Creates a Device-object to interface with a TMSI measurement system.

        Args:
            dev_type : <DeviceType> The measurement-system type.
                       Momentarily only the SAGA system is supported.

            dr_interface : <DeviceInterfaceType> The interface-type between the
                           data-recorder and docking-station (if the system exists
                           out of a DS and DR like the SAGA-system) or a PC.
                           The default interface-type = DeviceInterfaceType.docked

            ds_interface : <DeviceInterfaceType> The interface-type between the
                           docking station and PC.
                           The default interface-type = DeviceInterfaceType.usb

        Returns:
            <Device> An object of the system-implementation of the <Device-class>-interface.
            With this object one can interface with the attached system.
    """
    dev = None
    if (dev_type == DeviceType.saga):
        dev = SagaDevice(ds_interface, dr_interface)
    else:
        raise TMSiError(TMSiErrorCode.api_incorrect_argument)

    return dev

def discover(dev_type, dr_interface, ds_interface = DeviceInterfaceType.none):
    """Creates a list with Device-objects to interface with found TMSI measurement systems.

    Args:
        dev_type : <DeviceType> The measurement-system type.
                   Momentarily only the SAGA system is supported.

        dr_interface : <DeviceInterfaceType> The interface-type between the
                       data-recorder and docking-station (if the system exists
                       out of a DS and DR like the SAGA-system) or a PC.
                       The default interface-type = DeviceInterfaceType.docked

        ds_interface : <DeviceInterfaceType> The interface-type between the
                       docking station and PC.
                       The default interface-type = DeviceInterfaceType.usb

    Returns:
        <Device[]> An array with objects of the system-implementation of the
        <Device-class>-interface. With these object one can interface with the
        attached systems.
    """
    discoveryList = []

    if (dev_type == DeviceType.saga):
        discoveryList = saga_device.discover(ds_interface, dr_interface)

    return discoveryList