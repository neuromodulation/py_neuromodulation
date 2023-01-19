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
 * @file ${device.py} 
 * @brief Main Device Interface
 *
 */


'''

from enum import Enum, unique

@unique
class DeviceInterfaceType(Enum):
    none = 0
    usb = 1
    network = 2
    wifi = 3
    docked = 4
    optical = 5
    bluetooth = 6

@unique
class DeviceState(Enum):
    disconnected = 0
    connected = 1
    sampling = 2

@unique
class ChannelType(Enum):
    unknown = 0
    UNI = 1
    BIP = 2
    AUX = 3
    sensor = 4
    status = 5
    counter = 6
    all_types = 7

@unique
class MeasurementType(Enum):
    normal = 0
    impedance = 1

@unique
class ReferenceMethod(Enum):
    common = 0
    average = 1

@unique
class ReferenceSwitch(Enum):
    fixed=0
    auto=1

class DeviceChannel:
    """ 'DeviceChannel' represents a device channel. It has the next properties:

        type : 'ChannelType' Indicates the group-type of a channel.

        sample_rate : 'int' The curring sampling rate of the channel during a 'normal' measurement.
        
        bandwidth : 'int'  Bandwidth (in bits/s) required for transfer from DR to DS, 
                           used by bandwidth manager in application software.

        name : 'string' The name of the channel.

        unit_name : 'string' The name of the unit (e.g. 'μVolt)  of the sample-data of the channel.

        enabled : 'bool' Indicates if a channel is enabled for measuring

        Note: The properties 'name' and 'enabled' can be modified. The other properties
              are read-only.
    """

    def __init__(self, type, sample_rate, name, unit_name, enabled, bandwidth = 0, sensor = None):
        self.__type = type
        self.__sample_rate = sample_rate
        self.__unit_name = unit_name
        self.__name = name
        self.__enabled = enabled
        self.__sensor = sensor
        self.__bandwidth = bandwidth

    @property
    def type(self):
        """'ChannelType' Indicates the group-type of a channel."""
        return(self.__type)

    @property
    def sample_rate(self):
        """'int' The curring sampling rate of the channel during a 'normal' measurement."""
        return(self.__sample_rate)
        
    @property
    def bandwidth(self):
        """'int' Bandwidth (in bits/s) required for transfer from DR to DS, used by bandwidth manager in application software."""
        return(self.__bandwidth)

    @property
    def name(self):
        """'string' The name of the channel."""
        return self.__name

    @name.setter
    def name(self, var):
        """'string' Sets the name of the channel."""
        self.__name = var

    @property
    def unit_name(self):
        """'string' The name of the unit (e.g. 'μVolt)  of the sample-data of the channel."""
        return self.__unit_name

    @property
    def enabled(self):
        """'bool' Indicates if a channel is enabled for measuring"""
        return self.__enabled

    @enabled.setter
    def enabled(self, var):
        """'bool' Enables (True) or disables (False) a channel for measurement"""
        self.__enabled = var

    @property
    def sensor(self):
        """'DeviceSensor' contains an object if a sensor is attached to the channel."""
        return(self.__sensor)

class DeviceInfo:
    """ 'DeviceInfo' holds the static device information. It has the next properties:

        ds_interface : 'DeviceInterfaceType' Indicates interface-type between the PC
                       and the docking station.

        dr_interface : 'DeviceInterfaceType' Indicates interface-type between the
                       PC / docking station and the data recorder.

        ds_serial_number : 'int' The serial number of the docking station.

        dr_serial_number : 'int' The serial number of the data recorder.

    """
    def __init__(self, ds_interface = DeviceInterfaceType.none, dr_interface = DeviceInterfaceType.none):
        self.__ds_interface = ds_interface
        self.__dr_interface = dr_interface
        self.__ds_serial_number = 0
        self.__dr_serial_number = 0

    @property
    def ds_interface(self):
        """ds_interface : 'DeviceInterfaceType' Indicates interface-type between the PC
                          and the docking station.
        """
        return self.__ds_interface

    @property
    def dr_interface(self):
        """dr_interface : 'DeviceInterfaceType' Indicates interface-type between the
                       PC / docking station and the data recorder.
        """
        return self.__dr_interface

    @property
    def ds_serial_number(self):
        """'int' The serial number of the docking station."""
        return self.__ds_serial_number

    @property
    def dr_serial_number(self):
        """'int' The serial number of the data recorder."""
        return self.__dr_serial_number

class DeviceStatus:
    """ 'DeviceStatus' holds the runtime device information. It has the next properties:

        state : 'DeviceState' Indicates the connection-state towards the device which can be :

                    - connected : The connection to the device is open
                    - disconnected : The connection to the device is closed
                    - sampling : The connection to the device is open and the device
                                  has an ongoing measurement active.

        error : 'int' The return-code of the last made call to the device. A '0'
                indicates that the call was succesfull.
    """

    def __init__(self, state, error):
        self.__state = state
        self.__error = error

    @property
    def state(self):
        """'DeviceState' Indicates the connection-state towards the device"""
        return self.__state

    @property
    def error(self):
        """'int' The return-code of the last made call to the device."""
        return self.__error

class DeviceSensor:
    """ 'DeviceSensor' represents the sensor-data of a channel. It has the next properties:

    channel_list_idx : 'int' Index of the channel in the total channel list. Index starts at 0.
                        The total channel list can be accessed with [Device.config.channels]

    id : 'int' ID-code of the sensor used to identify BIP-sensors.

    serial_nr : 'int' Serial number of an attached AUX-sensor.

    product_id : 'int' Product identified of an attached AUX-sensor.

    exp :  'int' Exponent for the unit, e.g. milli = -3 this gives for a
            unit_name V a result mV.

    name : 'string' The name of the sensor-channel.

    unit_name : 'string' The name of the unit (e.g. 'Volt)  of the sensor-data.

    """

    def __init__(self, channel_list_idx, id, serial_nr, product_id, name, unit_name, exp):
        self.__channel_list_idx = channel_list_idx
        self.__id = id
        self.__serial_nr = serial_nr
        self.__name = name
        self.__unit_name = unit_name
        self.__product_id = product_id
        self.__exp = exp

    @property
    def channel_list_idx(self):
        """''int' Index of the channel in the total channel list."""
        return(self.__channel_list_idx)

    @property
    def id(self):
        """'int' ID-code of the sensor used to identify BIP-sensors."""
        return(self.__id)

    @property
    def serial_nr(self):
        """'int' Serial number of an attached AUX-sensor"""
        return self.__serial_nr

    @property
    def product_id(self):
        """'int' Product identified of an attached AUX-sensor"""
        return self.__product_id

    @property
    def name(self):
        """'string' The name of the sensor-channel."""
        return self.__name

    @property
    def unit_name(self):
        """'string' Exponent for the unit, e.g. milli = -3 this gives for a
            unit_name V a result mV.
        """
        return self.__unit_name

    @property
    def exp(self):
        """'int' The name of the channel."""
        return self.__exp


class DeviceConfig:
    """'DeviceConfig' holds the actual device configuration"""

    def __init__(self, sample_rate, num_channels):
        pass

    @property
    def num_channels(self):
        """'int' The total number of channels (enabled and disabled)"""
        pass

    @property
    def channels(self):
        """'list DeviceChannel' The total list of channels (enabled and disabled)"""
        pass

    @channels.setter
    def channels(self, ch_list):
        """Updates the channel lists in the device.

        Args:
            ch_list : 'list DeviceChannel'

            The channel-properties 'enabled' and 'name' can be modified. The other
            channel-properties are read-only.

        Note:
            'ch_list' must contain  all channels (enabled and disabled), in the
            same sequence as retrieved with 'Device.config.channels'

            It is advised to "refresh" the applications' local "variables"
            after the channel list has been updated.
        """
        pass

    @property
    def base_sample_rate(self):
        """'int' The active base-sample-rate of the device"""
        pass

    @base_sample_rate.setter
    def base_sample_rate(self, sample_rate):
        """Sets the base-sample-rate of the device.

        Args:
            sample_rate : 'int' new base sample rate

        Note:
            Upon a change of the base-sample-rate, automatically the sample-rate
            of the channel-type-groups (and thus also the sample-rate of every channel)
            changes.

            It is advised to "refresh" the applications' local "variables"
            after the base-sample-rate has been updated.
        """
        pass

    @property
    def sample_rate(self):
        """'int' The rate of the current configuration, with which
            sample-sets are sent during a measurement.
        """
        pass

    def get_sample_rate(self, channel_type):
        """'int' the sample-rate of the specified channel-type-group.

        Args:
            channel_type : 'ChannelType' The channel-type-group.
        """
        pass

    def set_sample_rate(self, channel_type, base_sample_rate_divider):
        """Sets the sample-rate of the specified channel-type-group.

        Args:
            channel_type : 'ChannelType' The channel-type-group of which the
            sample-rate must be updated.
            It is possible to set all channels to the same sample-rate within one
            call. Then 'ChannelType.all_types' must be used.

            base_sample_rate_divider: 'int' The divider to indicate to what fraction
            of the active base-sample-rate the sample-rate of the channel-type-group
            must be set. Only the values 1, 2, 4 and 8 are possible.
            For example if the base-sample-rate=4000 and base_sample_rate_divider=8,
            the sample-rate will become 4000/8 = 500 Hz

        Note:
            Upon a change of the sample-rate of a channel-type-group, automatically
            the sample-rate of the channels of that channel-type are updated.

            It is advised to "refresh" the applications' local "variables"
            after the sample-rate of a channel-type-group has been updated.
        """
        pass
    
    def set_interface_type(self, dr_interface_type):
        """Changes the configured interface type of the device
        
        Args:
            interface_type: 'DeviceInterfaceType' The interface type that needs 
            to be updated.
        """
        pass
        
    @property
    def interface_bandwidth(self):
        """'int' Data bandwidth in Mbits/s for current interface."""
        pass
    
    @property
    def reference_method(self):
        """ 'ReferenceMethod' the reference method applied to the UNI channels"""
        pass
    
    @reference_method.setter
    def reference_method(self, reference_type):
        """Sets the reference method for the UNI channels
        
        Args:
            reference_type: 'ReferenceMethod' the type of reference method that
            should be used for measuring the UNI channels
        
        """
    
class Device:
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

    @property
    def id(self):
        """ 'int' : Unique id within all available devices. The id can be used to
            register as a client at the 'sample_data_server' for retrieval of
            sample-data of a specific device
        """
        pass

    @property
    def info(self):
        """ 'class DeviceInfo' : Static information of a device like used interfaces, serial numbers
        """
        pass

    @property
    def status(self):
        """ 'class DeviceStatus' : Runtime information of a device like device state
        """
        pass

    @property
    def config(self):
        """ 'class DeviceConfig' : The configuration of a device which exists
            out of individual properties (like base-sample-rate) and the total
            channel list (with enabled and disabled channels)
        """
        pass

    @property
    def channels(self):
        """ 'list of class DeviceChannel' : The list of enabled channels.
            Enabled channels are active during an 'normal' measurement.
        """
        pass

    @property
    def imp_channels(self):
        """ 'list of class DeviceChannel' : The list of impedance channels.
            Impedance channels are active during an 'impedance' measurement.
        """
        pass

    @property
    def sensors(self):
        """ 'list of class DeviceSensor' : The complete list of sensor-information
            for the  sensor-type channels : BIP and AUX
        """

    @property
    def datetime(self):
        """ 'datetime' Current date and time of the device
        """
        pass

    @datetime.setter
    def datetime(self, dt):
        """ 'datetime' Sets date and time of the device
        """
        pass

    def open(self):
        """ Opens the connection to the device.

            The open-function will first initiate a discovery on attached systems to the PC
            based on the interface-types which were registered upon the creation of the Device-object.

            A connection will be established with the first available system.

            The functionailty a device offers will only be available when a connection
            to the system has been established.
        """
        pass

    def close(self):
        """ Closes the connection to the device.
        """
        pass

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
        pass

    def stop_measurement(self):
        """ Stops an ongoing measurement on the device."""
        pass

    def set_factory_defaults(self):
        """ Initiates a factory reset to restore the systems' default configuration."""
        pass

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
        pass

    def save_config(self, filename):
        """ Saves the current device configuration to file.

        Args:
            filename : path and filename of the file to which the current
                        device configuration must be saved.
        """
        pass

    def update_sensors(self):
        """ Called when sensors have been attached or detached to/from the device.
            The complete configuration including the new sensor-configuration
            is reloaded from the device.

        Note:
            It is advised to "refresh" the applications' local "variables"
            after the the complete configuration has been reloaded.
        """
        pass

