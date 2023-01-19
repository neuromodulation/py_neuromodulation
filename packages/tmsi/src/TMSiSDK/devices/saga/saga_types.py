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
 * @file ${saga_types.py} 
 * @brief SAGA Device Types.
 *
 */


'''

from ...device import DeviceInterfaceType, DeviceState, DeviceConfig, ChannelType, DeviceChannel, DeviceSensor, ReferenceMethod, ReferenceSwitch

_TMSI_DEVICE_ID_NONE = 0xFFFF

class SagaConst:
    TMSI_DEVICE_ID_NONE = 0xFFFF

class SagaInfo():
    """ 'DeviceInfo' holds the static device information. It has the next properties:

        ds_interface : 'DeviceInterfaceType' Indicates interface-type between the PC
                       and the docking station.

        dr_interface : 'DeviceInterfaceType' Indicates interface-type between the
                       PC / docking station and the data recorder.

        ds_serial_number : 'int' The serial number of the docking station.

        dr_serial_number : 'int' The serial number of the data recorder.

    """
    def __init__(self, ds_interface = DeviceInterfaceType.none, dr_interface = DeviceInterfaceType.none):
        self.ds_interface = ds_interface
        self.dr_interface = dr_interface
        self.id = _TMSI_DEVICE_ID_NONE;
        self.state = DeviceState.disconnected

class SagaConfig(DeviceConfig):
    """'DeviceConfig' holds the actual device configuration"""
    def __init__(self):
        self._parent = None
        self._base_sample_rate = 0 # base sample reate in Hz
        self._configured_interface = DeviceInterfaceType.none
        self._triggers = 0 # 0= Disabled, 1= external triggers enabled
        self._reference_method = 0 # 0= Common reference, 1=average reference
        self._auto_reference_method = 0 #  0= fixed method, 1= autoswitch to average reference when CREF is out of range
        self._dr_sync_out_divider = -1 # SetBaseSampleRateHz/SyncOutDiv, -1 = marker button
        self._dr_sync_out_duty_cycle = 0 # DR Sync dutycycle
        self._repair_logging = 0 # 0 = Disabled, 1 = BackupLogging enabled,
        self._num_channels = 0; # Total number of channels : active and inactive
        self._channels = [] # Total channel list : active and inactive
        self._sample_rates = [] # List with sample_rate per channel-type
        self._num_sensors = 0 # Number of sensors
        self._interface_bandwidth = 0; # Data bandwidth in Mbits/s for current interface.
        for chan_type in ChannelType:
            self._sample_rates.append(SagaSampleRate(chan_type))

    def get_sample_rate(self, chan_type):
        """'int' the sample-rate of the specified channel-type-group.

        Args:
            channel_type : 'ChannelType' The channel-type-group.
        """
        return self._sample_rates[chan_type.value].sample_rate

    def set_sample_rate(self, chan_type, bsr_div):
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
        if (bsr_div == 1) or (bsr_div == 2) or (bsr_div == 4) or (bsr_div == 8):
            bsr_shift = 0
            while (bsr_div > 1):
                bsr_shift += 1
                bsr_div /= 2
            for ch in self._channels:
                if ((ch.type == chan_type) or (chan_type == ChannelType.all_types) and (ch.chan_divider != -1)):
                    # Only update the chan_divider of active channels
                    if (ch.chan_divider != -1):
                        ch.chan_divider = bsr_shift
            if (self._parent != None):
                self._parent._update_config()
        else:
            print('\nProvided base_sample_rate_divider is invalid. Sample rate can not be updated.\n')

    @property
    def base_sample_rate(self):
        """'int' The active base-sample-rate of the device"""
        return self._base_sample_rate

    @base_sample_rate.setter
    def base_sample_rate(self, var):
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
        if (var == 4000) or (var == 4096):
            if self._base_sample_rate != var:
                # The device does not support a configuration with different base 
                # sample rates for sample rate and sync out. Hence sync out config
                # has to be update as well
                sync_out_freq=(self._base_sample_rate/self._dr_sync_out_divider)
                self._base_sample_rate = var
                self.set_sync_out_config(freq=sync_out_freq)
        else:
            print('\nProvided base_sample_rate is invalid. Sample rate can not be updated.\n')

    @property
    def sample_rate(self):
        """<int> The rate of the current configuration, with which
            sample-sets are sent during a measurement. This always
            equals the sample-rate of the COUNTER-channel.
        """
        return self._sample_rates[ChannelType.counter.value].sample_rate

    @property
    def configured_interface(self):
        return self._configured_interface
    
    def set_interface_type(self, dr_interface_type):
        """Changes the configured interface type of the device.
        
        Args:
            dr_interface_type: 'DeviceInterfaceType' The interface type that needs 
            to be updated.
            
        Note:
            The interface type switch applies to the DR-DS interface type.
            Interface types that can be configured on SAGA are docked, 
            optical and wifi.
        """
        
        if dr_interface_type.value != self._configured_interface:
            print('DR-DS interface is changed to: ')
            print(dr_interface_type)
            self._configured_interface = dr_interface_type.value
        
            if (self._parent != None):
                self._parent._update_config()

    @property
    def num_channels(self):
        """'int' The total number of channels (enabled and disabled)"""
        return self._num_channels

    @property
    def channels(self):
        """'list DeviceChannel' The total list of channels (enabled and disabled)"""
        chan_list = []
        for ch in self._channels:
            sensor = ch.sensor
            if (ch.sensor != None):
                sensor = DeviceSensor(ch.sensor.idx_total_channel_list,
                                      ch.sensor.id,
                                      ch.sensor.serial_nr,
                                      ch.sensor.product_id,
                                      ch.sensor.name,
                                      ch.sensor.unit_name,
                                      ch.sensor.exp)

            dev_ch = DeviceChannel(ch.type, ch.sample_rate, ch.alt_name, ch.unit_name, (ch.chan_divider != -1), sensor)
            chan_list.append(dev_ch)
        return chan_list

    @channels.setter
    def channels(self, var):
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
        for idx, channel in enumerate(var):
            if (self._channels[idx].type.value < ChannelType.status.value):
                self._channels[idx].alt_name = channel.name
                if (channel.enabled == True):
                    if (self._sample_rates[self._channels[idx].type.value].sample_rate != 0):
                        self._channels[idx].sample_rate = self._sample_rates[self._channels[idx].type.value].sample_rate
                        self._channels[idx].chan_divider = self._sample_rates[self._channels[idx].type.value].chan_divider
                    else:
                        self._sample_rates[self._channels[idx].type.value].sample_rate = self._sample_rates[ChannelType.counter.value].sample_rate
                        self._sample_rates[self._channels[idx].type.value].chan_divider = self._sample_rates[ChannelType.counter.value].chan_divider
                        self._channels[idx].sample_rate = self._sample_rates[self._channels[idx].type.value].sample_rate
                        self._channels[idx].chan_divider = self._sample_rates[self._channels[idx].type.value].chan_divider
                else:
                    self._channels[idx].chan_divider = -1
        if (self._parent != None):
            self._parent._update_config()
            
    @property
    def reference_method(self):
        """ 'ReferenceMethod' the reference method applied to the UNI channels,
        'ReferenceSwitch' the switching of reference mode when common reference is disconnected"""
        return ReferenceMethod(self._reference_method).name,  ReferenceSwitch(self._auto_reference_method).name
    
    @reference_method.setter
    def reference_method(self, reference_type):
        """Sets the reference method for the UNI channels
        
        Args:
            reference_type: 'ReferenceMethod' the type of reference method that
            should be used for measuring the UNI channels
            'ReferenceSwitch' the switching of reference mode when common reference is disconnected
        
        """
        if not type(reference_type) is tuple:
            if isinstance(reference_type, ReferenceMethod):
                self._reference_method=reference_type.value
            elif isinstance(reference_type, ReferenceSwitch):
                self._auto_reference_method=reference_type.value
               
        else:
            for ind in range(len(reference_type)):
                if isinstance(reference_type[ind], ReferenceMethod):
                    self._reference_method=reference_type[ind].value
                elif isinstance(reference_type[ind], ReferenceSwitch):
                    self._auto_reference_method=reference_type[ind].value

        if (self._parent != None):
            self._parent._update_config()
            
    @property
    def triggers(self):
        """'Boolean', true when triggers are enabled or false when disabled"""
        return bool(self._triggers)
    
    @triggers.setter
    def triggers(self, enable_triggers):
        """Sets the triggers to enabled or disabled"""
        self._triggers=enable_triggers
        if (self._parent != None):
            self._parent._update_config()
        
    @property
    def repair_logging(self):
        """Boolean, true in case samples are logged on Recorder for repair in case of data loss"""
        return self._repair_logging
    
    @repair_logging.setter
    def repair_logging(self, enable_logging):
        """Sets repair logging to enabled or disabled"""
        self._repair_logging=enable_logging
        if (self._parent != None):
            self._parent._update_config()
    
    def get_sync_out_config(self):
        """Sync out configuration, shows whether sync out is in marker mode or square wave mode with corresponding frequency and duty cycle"""
        if self._dr_sync_out_divider==-1:
            freq=-1
        else:
            freq=(self._base_sample_rate/self._dr_sync_out_divider)
        return self._dr_sync_out_divider==-1, freq, self._dr_sync_out_duty_cycle/10
    
    def set_sync_out_config(self, marker=False, freq=None, duty_cycle=None):
        """Set sync out to marker mode or square wave mode with corresponding frequency and duty cycle"""
        if marker:
            self._dr_sync_out_divider=-1
        else:
            if freq:
                self._dr_sync_out_divider=round(self._base_sample_rate/freq)
            if duty_cycle:
                self._dr_sync_out_duty_cycle=duty_cycle*10
                
        if (self._parent != None):
            self._parent._update_config()

    @property
    def interface_bandwidth(self):
        return self._interface_bandwidth
        
    @interface_bandwidth.setter
    def interface_bandwidth(self, bandwidth):
        """Sets the data bandwidth in Mbits/s for current interface."""
        self._interface_bandwidth=bandwidth      

class SagaSensor():
    """ <SagaSensor> represents the sensor-data of a channel. It has the next properties:

    offset : <float> Offset for the seonsor-channel.

    gain : <float> Value to convert the sensor value to the correct unit value

    exp :  <int> Exponent for the unit, e.g. milli = -3 this gives for a
            unit_name V a result mV.

    name : <string> The name of the seonsor-channel.

    unit_name : <string> The name of the unit (e.g. 'μVolt)  of the sensor-data.

    """
    def __init__(self):
        self.idx_total_channel_list = -1
        self.id = -1
        self.manufacturer_id = 0
        self.serial_nr = 0
        self.product_id = 0
        self.offset = 0
        self.gain =1
        self.exp = 0
        self.__name = ""
        self.__unit_name = ""

    @property
    def name(self):
        """'string' The name of the unit (e.g. 'μVolt)  of the sample-data of the channel."""
        return self.__name

    @name.setter
    def name(self, bname):
        self.__name = ""
        new_name = bytearray()
        for i in range (len(bname)):
            if bname[i] > 127:
                new_name.append(194) #0xC2
            new_name.append(bname[i])
        if len(new_name) > 0:
            self.__name = new_name.decode('utf8').rstrip('\x00')

    @property
    def unit_name(self):
        """'string' The name of the unit (e.g. 'μVolt)  of the sample-data of the channel."""
        return self.__unit_name

    @unit_name.setter
    def unit_name(self, bname):
        # A unit-name can start with the micro-character (0xB5). In that case must
        # the micro-character be converted to it;s utf8-representation : 0xC2B5
        self.__unit_name = ""
        new_name = bytearray()
        for i in range (len(bname)):
            if bname[i] > 127:
                new_name.append(194) #0xC2
            new_name.append(bname[i])
        if len(new_name) > 0:
            self.__unit_name = new_name.decode('utf8').rstrip('\x00')

class SagaChannel():
    """ <DeviceChannel> represents a device channel. It has the next properties:

    type : <ChannelType> Indicates the group-type of a channel.

    sample_rate : <int> The curring sampling rate of the channel during a 'normal' measurement.

    name : <string> The name of the channel.

    unit_name : <string> The name of the unit (e.g. 'μVolt)  of the sample-data of the channel.

    enabled : <bool> Indicates if a channel is enavled for measuring

    Note: The properties <name> and <anabled> can be modified. The other properties
		  are read-only.
    """
    def __init__(self):
        self.type = 0
        self.format = 0
        self.sample_rate = 0
        self.chan_divider = -1
        self.imp_divider = -1;
        self.exp = 1
        self.__unit_name = "-"
        self.def_name = "-"
        self.alt_name = "-"
        self.sensor = None

    @property
    def unit_name(self):
        """'string' The name of the unit (e.g. 'μVolt)  of the sample-data of the channel."""
        return self.__unit_name

    @unit_name.setter
    def unit_name(self, name):
        # A unit-name can start with the micro-character (0xB5). In that case must
        # the micro-character be converted to it;s utf8-representation : 0xC2B5
        self.__unit_name = ""
        bname = name.encode('windows-1252')
        new_name = bytearray()
        for i in range (len(bname)):
            if bname[i] > 127:
                new_name.append(194) #0xC2
            new_name.append(bname[i])
        if len(new_name) > 0:
            self.__unit_name = new_name.decode('utf8')

class SagaSampleRate():
    def __init__(self, type):
        self.type = type
        self.sample_rate = 0
        self.chan_divider = -1