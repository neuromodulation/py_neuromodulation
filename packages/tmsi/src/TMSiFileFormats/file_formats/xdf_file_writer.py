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
 * @file ${xdf_file_writer.py} 
 * @brief XDF File Writer Interface
 *
 */


'''

from datetime import datetime, timedelta
from enum import IntEnum

import threading
import queue
import struct
import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os

from TMSiSDK.device import ChannelType
from TMSiSDK.error import TMSiError, TMSiErrorCode
from TMSiSDK import sample_data_server

from apex_sdk.device.tmsi_device import TMSiDevice
from apex_sdk.sample_data_server.sample_data_server import SampleDataServer as ApexSampleDataServer 


from os.path import join, dirname, realpath
Writer_dir = dirname(realpath(__file__)) # directory of this file
measurements_dir = join(Writer_dir, '../../measurements') # directory with all measurements
modules_dir = join(Writer_dir, '../../') # directory with all modules

_QUEUE_SIZE_SAMPLE_SETS = 1000


class ChunkTag(IntEnum):
    """ <ChunkTag> The chunk tag defines the type of the chunk."""
    file_header = 1
    stream_header = 2
    samples = 3
    clock_offset = 4
    boundary = 5
    stream_footer = 6


def xml_etree_to_string(elem):
    """Returns a XML string for the XML Element.

        Args:
            elem : 'ET.Element'
    """
    rough_string = ET.tostring(elem, 'utf-8')
    return rough_string

class XdfWriter:
    def __init__(self, filename, add_ch_locs):
        self.q_sample_sets = queue.Queue(_QUEUE_SIZE_SAMPLE_SETS)
        self.device = None

        self.filename=filename    
        self._fp = None
        self._date = None
        self.add_ch_locs=add_ch_locs

    def open(self, device):
        """ Opens and initializes a xdf file-writer session.

            1. Opens the xdf-file
            2. Writes the magic code 'XDF:'
            3. Writes the FileHeader-chunk
            4. Determines the number of sample sets within one Samples-chunk
            5. Registers at the sample-data-server and start the sampling-thread
        """
        print("XdfWriter-open")
        if isinstance(device, TMSiDevice):
            self.__open_TMSiDevice(device)
            return
        
        self.device = device
        
        self._sample_rate = device.config.sample_rate
        self._num_channels = len(device.channels)
        
        now = datetime.now()
        self._date = now
        filetime = now.strftime("%Y%m%d_%H%M%S")
        fileparts=self.filename.split('.')
        if fileparts[-1]=='xdf' or fileparts[-1]=='Xdf':
            self.filename='.'.join(fileparts[:-1])+ '-' + filetime + '.xdf'
        else:
            self.filename = self.filename + '-' + filetime + '.xdf'
        
        #Check for recent impedance values
        imp_df = None
        for file in os.listdir(measurements_dir):
            if ('.txt' in file) and ('Impedances_' in file):
                Impedance_time = datetime.strptime(file[-19:-4], "%Y%m%d_%H%M%S")
                if (now-Impedance_time) < timedelta(minutes=2):
                    imp_file = file
                    #read impedance data
                    imp_df = pd.read_csv(join(measurements_dir,file), delimiter = "\t", header = None)
                    imp_df.columns = ['ch_name', 'impedance', 'unit']
        if imp_df is not None:
             print('Included impedance values from file:', imp_file)
             
        try:
            # 1. Open the xdf-file
            self._fp = open(self.filename, 'wb')

            # 2. Write the magic code 'XDF:'
            self._fp.write(str.encode("XDF:"))
            self._write_file_header_chunk()

            # 3. Write the file-header chunk
            self.device.config
          
            self._write_stream_header_chunk(self.device.channels, self._sample_rate, imp_df)
                
            # 4. Determine the number of sample-sets within one Samples-chunk:
            #   This is the number of sample-sets received within 150 milli-seconds or when the
            #   sample-data-block-size exceeds 64kb it will become the number of sample-sets that fit within 64kb
            self._num_sample_sets_per_sample_data_block = int(self._sample_rate * 0.15)
            size_one_sample_set = len(self.device.channels) * 4
            if ((self._num_sample_sets_per_sample_data_block * size_one_sample_set) > 64000):
                 self._num_sample_sets_per_sample_data_block = int(64000 / size_one_sample_set)
                 
            fmt = 'f'*self._num_channels
            self.pack_struct = struct.Struct(fmt)     

            # 5. Register at the sample-data-server and start the sampling-thread
            sample_data_server.registerConsumer(self.device.id, self.q_sample_sets)
            self._sampling_thread = ConsumerThread(self, name='Xdf-writer : dev-id-' + str(self.device.id))
            self._sampling_thread.start()
        except:
            raise TMSiError(TMSiErrorCode.file_writer_error)
        
            
    def __open_TMSiDevice(self, device):          
        """ Opens and initializes a xdf file-writer session.

            1. Opens the xdf-file
            2. Writes the magic code 'XDF:'
            3. Writes the FileHeader-chunk
            4. Determines the number of sample sets within one Samples-chunk
            5. Registers at the sample-data-server and start the sampling-thread
        """
        
        self.device = device
        
        self._sample_rate = self.device.get_device_sampling_frequency()
        self._num_channels = self.device.get_num_channels()
        
        now = datetime.now()
        self._date = now
        filetime = now.strftime("%Y%m%d_%H%M%S")
        fileparts = self.filename.split('.')
        if fileparts[-1] == 'xdf' or fileparts[-1] == 'Xdf':
            self.filename='.'.join(fileparts[:-1])+ '-' + filetime + '.xdf'
        else:
            self.filename = self.filename + '-' + filetime + '.xdf'
        
        #Check for recent impedance values
        imp_df = None
        for file in os.listdir(measurements_dir):
            if ('.txt' in file) and ('Impedances_' in file):
                Impedance_time = datetime.strptime(file[-19:-4], "%Y%m%d_%H%M%S")
                if (now-Impedance_time) < timedelta(minutes=2):
                    imp_file = file
                    #read impedance data
                    imp_df = pd.read_csv(join(measurements_dir,file), delimiter = "\t", header = None)
                    imp_df.columns = ['ch_name', 'impedance', 'unit']
        if imp_df is not None:
              print('Included impedance values from file:', imp_file)
             
        try:
            # 1. Open the xdf-file
            self._fp = open(self.filename, 'wb')

            # 2. Write the magic code 'XDF:'
            self._fp.write(str.encode("XDF:"))
            self._write_file_header_chunk()

            # 3. Write the file-header chunk
            # self.device.config
          
            channels = self.device.get_device_active_channels()
            self._write_stream_header_chunk(channels, self._sample_rate, imp_df)
                
            # 4. Determine the number of sample-sets within one Samples-chunk:
            #   This is the number of sample-sets received within 150 milli-seconds or when the
            #   sample-data-block-size exceeds 64kb it will become the number of sample-sets that fit within 64kb
            self._num_sample_sets_per_sample_data_block = int(self._sample_rate * 0.15)
            size_one_sample_set = self._num_channels * 4
            if ((self._num_sample_sets_per_sample_data_block * size_one_sample_set) > 64000):
                 self._num_sample_sets_per_sample_data_block = int(64000 / size_one_sample_set)
                 
            fmt = 'f'*self._num_channels
            self.pack_struct = struct.Struct(fmt)     

            # 5. Register at the sample-data-server and start the sampling-thread
            ApexSampleDataServer().register_consumer(self.device.get_id(), self.q_sample_sets)
            self._sampling_thread = ConsumerThread(self, name='Xdf-writer : dev-id-' + str(self.device.get_id()))
            self._sampling_thread.start()
        except:
            raise TMSiError(TMSiErrorCode.file_writer_error)
            
            

    def save_offline(self, stream_info, streams):
        """ Opens and initializes a xdf file-writer session for an offline processing.

            1. Opens the xdf-file
            2. Writes the magic code 'XDF:'
            3. Writes the FileHeader-chunk
            4. Writes the Stream header chunk
            5. Saves all data 
        """
        try:
            # 1. Open the xdf-file
            self._fp = open(self.filename, 'wb')
            # 2. Write the magic code 'XDF:'
            self._fp.write(str.encode("XDF:"))
            # 3. Write the file-header chunk
            self._write_file_header_chunk()
            # 4. Write the stream-header chunk
            self._write_stream_header_chunk_offline(stream_info)
            #5 write data
            self._write_data_streams_into_file(streams)
            #6 write footer
            self._write_stream_footer_chunk(0, 10, self._num_written_sample_sets, self._sample_rate)
            #7 close
            self._fp.close()

        except:
            raise TMSiError(TMSiErrorCode.file_writer_error)
    
    def _write_data_streams_into_file(self, streams):
        """Method that writes the streams for offline saving of data streams"""
        n_ch = len(streams)
        fmt = 'f'*n_ch
        self.pack_struct = struct.Struct(fmt)  
        _num_sample_sets_per_sample_data_block = int(6400000 / 4 / n_ch)
        self._num_written_sample_sets = 0
        _sample_set_block_index = 0
        _boundary_chunk_counter = 0
        _boundary_chunk_counter_threshold = 10 * self._sample_rate
        n_samp = int(len(streams[0]) * len(streams) / n_ch)
        n_iter = np.int(np.floor(n_samp/_num_sample_sets_per_sample_data_block))
        try:
            for i in range(n_iter):
                time_range = [j for j in range(i*_num_sample_sets_per_sample_data_block,(i+1)*_num_sample_sets_per_sample_data_block)]
                self._sample_sets_in_block = [streams[n_channel][n_sample] for n_sample in time_range for n_channel in range(len(streams))]
                print("\rwriting progress: {:.2f}%\r".format(100*i/n_iter), end="\r")
                XdfWriter._write_sample_chunk(self._fp,\
                                            self._sample_sets_in_block,\
                                            _num_sample_sets_per_sample_data_block, \
                                            n_chan=n_ch, \
                                            pack_struct=self.pack_struct)
            
                self._num_written_sample_sets += _num_sample_sets_per_sample_data_block
                _sample_set_block_index += 1

    
                # Write approximately every 10 seconds a Boundary-chunk
                _boundary_chunk_counter += _num_sample_sets_per_sample_data_block
                if (_boundary_chunk_counter >= _boundary_chunk_counter_threshold):
                    XdfWriter._write_boundary_chunk(self._fp)
                    _boundary_chunk_counter = 0
            
            # Store remaining samples for next repetion
            i = np.int(np.floor(n_samp/_num_sample_sets_per_sample_data_block))
            time_range = [j for j in range(i*_num_sample_sets_per_sample_data_block,n_samp)]
            self._sample_sets_in_block = [streams[n_channel][n_sample] for n_sample in time_range for n_channel in range(len(streams))]
            XdfWriter._write_sample_chunk(self._fp,\
                                        self._sample_sets_in_block,\
                                        n_samp-i*_num_sample_sets_per_sample_data_block, \
                                        n_chan=n_ch, \
                                        pack_struct=self.pack_struct)
            print("\rwriting progress: 100.00%") 

        except:
            raise TMSiError(TMSiErrorCode.file_writer_error)

    def close(self):
        """ Closes a xdf file-writer session.

            1. Stops the sampling-thread
            2. Writes the StreamFooter-chunk (by the sampling-thread)
            3. Closes the xdf-file (by the sampling-thread)
        """
        print("XdfWriter-close")
        self._sampling_thread.stop_sampling()
        
        if isinstance(self.device, TMSiDevice):
            ApexSampleDataServer().unregister_consumer(self.device.get_id(), self.q_sample_sets)
        else:
            sample_data_server.unregisterConsumer(self.device.id, self.q_sample_sets)


    @staticmethod
    def _write_chunk(f, length_size, chunk_tag, chunk_data):
        """ Writes a complete chunk to the xdf-file. Writes the chunk-meta-data and chunk-data.

            Args:
                f : 'file-object' of the xdf-file
                length_size : 'int' variable-length indicator of the chunk-length (1, 4 or 8)
                chunk_tag : 'ChunkTag' defines the type of the chunk
                chunk_data : byte-array with the chunk-data to write
        """
        # 1. Write the chunk meta-data :
        #       - chunk-length,
        #       - arbitary bytes
        #       - stream-id (when needed)
        #       - chunk-tag
        if (length_size == 1):
            length_size_type = b'\1'
        else:
            if (length_size == 4):
                length_size_type = b'\4'
            else:
                length_size_type = b'\8'

        num_arbitrary_bytes = 2
        data_size = len(chunk_data) + num_arbitrary_bytes
        if (chunk_tag != ChunkTag.file_header):
            data_size += 4

        f.write(length_size_type)
        f.write(data_size.to_bytes(length_size, 'little'))
        f.write(chunk_tag.to_bytes(num_arbitrary_bytes, 'little'))
        if (chunk_tag != ChunkTag.file_header):
            _stream_id = 1
            f.write(_stream_id.to_bytes(4, 'little'))

        # 2. Write the chunk-data
        f.write(chunk_data)

    def _write_file_header_chunk(self):
        """ Writes the standard FileHeader-chunk."""
        data = ET.Element('info')
        item = ET.SubElement(data, 'version')
        item.text = '1'
        XdfWriter._write_chunk(self._fp, 1, ChunkTag.file_header, xml_etree_to_string(data))

    def _write_stream_header_chunk_offline(self, stream_info):
        de_info = ET.Element("info")
        item_name = ET.SubElement(de_info, 'name')
        item_name.text = stream_info["name"][0]
        item_type = ET.SubElement(de_info, 'type')
        item_type.text = stream_info["type"][0]
        item_channel_count = ET.SubElement(de_info, 'channel_count')
        item_channel_count.text =  stream_info["channel_count"][0]
        item_nominal_srate = ET.SubElement(de_info, 'nominal_srate')
        item_nominal_srate.text =  stream_info["nominal_srate"][0]
        self._sample_rate = int(float(stream_info["nominal_srate"][0])) # needed for later
        item_channel_format = ET.SubElement(de_info, 'channel_format')
        item_channel_format.text = stream_info["channel_format"][0]
        # Assemble the EEG meta-data
        de_desc = ET.SubElement(de_info, 'desc')
        de_channels = ET.SubElement(de_desc, 'channels')
        for channel in stream_info["desc"][0]["channels"][0]["channel"]:
            item_channel = ET.SubElement(de_channels, 'channel')
            # channel label
            item_label =  ET.SubElement(item_channel, 'label')
            item_label.text = channel["label"][0]
            # channel index
            item_index =  ET.SubElement(item_channel, 'index')
            item_index.text = channel["index"][0]
            # channel content-type (EEG, EMG, EOG, ...)
            item_type = ET.SubElement(item_channel, 'type')
            item_type.text = channel["type"][0]
            # measurement unit (strongly preferred unit: microvolts)
            item_unit = ET.SubElement(item_channel, 'unit')
            item_unit.text = "-"
            if channel["impedance"]:
                item_impedance = ET.SubElement(item_channel, 'impedance')
                if len(channel["impedance"]) > 0:
                    item_impedance.text = channel["impedance"][0]
            
        
        item_manufacturer =  ET.SubElement(de_desc, 'manufacturer')
        item_manufacturer.text = stream_info["desc"][0]["manufacturer"][0]
        # Write the StreamHeader-cunk
        XdfWriter._write_chunk(self._fp, 4, ChunkTag.stream_header, xml_etree_to_string(de_info))


    def _write_stream_header_chunk(self, channels, sample_rate, imp_df=None):
        """ Writes the StreamHeader-chunk :
            - <info>-element with:
                - device-name,
                - overall content-type,
                - channel-count,
                - sample-rate and
                - sample-data-format
                - EEG meta-data, the channels-element ('<desc><channels>'):
                    - channel-name (<label>)
                    - channel-type (<type>)
                    - channel-unit-name (<unit>)
                    - channel-location (<location>)
                    - channel-impedance (<impedance>)
                -reference meta-data
                -acquistion meta-data

            Args:
                channels : 'list DeviceChannel' channel list
                sample_rate : int' The rate of the current configuration, with which
                                   sample-sets are sent during a measurement.
                imp_df: 'DataFrame' Previously recorded impedances 
        """
        
        if isinstance(self.device, TMSiDevice):
            self.__write_stream_header_chunk(channels, sample_rate, imp_df)
            return
        
        # Assemble <info>-root-element:
        num_channels = self._num_channels
        de_info = ET.Element('info')
        item_name = ET.SubElement(de_info, 'name')
        item_name.text = 'SAGA'
        item_type = ET.SubElement(de_info, 'type')
        item_type.text = 'EEG'
        item_channel_count = ET.SubElement(de_info, 'channel_count')
        item_channel_count.text =  str(num_channels)
        item_nominal_srate = ET.SubElement(de_info, 'nominal_srate')
        item_nominal_srate.text =  str(sample_rate)
        item_channel_format = ET.SubElement(de_info, 'channel_format')
        item_channel_format.text = 'float32'

        # Assemble the EEG meta-data
        de_desc = ET.SubElement(de_info, 'desc')
        de_channels = ET.SubElement(de_desc, 'channels')

        #read channel locations
        chLocs=pd.read_csv(join(modules_dir,'TMSiSDK/_resources/EEGchannelsTMSi3D.txt'), sep="\t", header=None)
        chLocs.columns=['default_name', 'eeg_name', 'X', 'Y', 'Z']

        # Meta-data per channel
        i=0  #active channel counter
        for j in range(len(self.device._config._channels)):
            if self.device._config._channels[j].def_name==self.device._channels[i].def_name:
                # description of one channel, repeated (one for each channel in the time series)
                item_channel = ET.SubElement(de_channels, 'channel')
                # channel label
                item_label =  ET.SubElement(item_channel, 'label')
                item_label.text = channels[i].name
                # channel content-type (EEG, EMG, EOG, ...)
                item_type = ET.SubElement(item_channel, 'type')
                
                if (channels[i].type.value == ChannelType.UNI.value):
                    if not j==0:
                        item_type.text = 'EEG'
                        #channel location
                        if self.add_ch_locs:
                            item_location = ET.SubElement(item_channel, 'location')
                            item_x = ET.SubElement(item_location, 'X')
                            item_y = ET.SubElement(item_location, 'Y')
                            item_z = ET.SubElement(item_location, 'Z')
                            item_x.text=str(95*chLocs['X'].values[j-1])
                            item_y.text=str(95*chLocs['Y'].values[j-1])
                            item_z.text=str(95*chLocs['Z'].values[j-1])
                    else:
                        item_type.text = 'CREF'
                elif (channels[i].type.value == ChannelType.BIP.value):
                    item_type.text = 'BIP'
                elif (channels[i].type.value == ChannelType.AUX.value):
                    item_type.text = 'AUX'
                elif (channels[i].type.value == ChannelType.sensor.value):
                    item_type.text = 'sensor'
                elif (channels[i].type.value == ChannelType.status.value):
                    item_type.text = 'status'
                elif (channels[i].type.value == ChannelType.counter.value):
                    item_type.text = 'counter'
                else:
                    item_type.text = '-'
                    
                if imp_df is not None:
                    #channel impedence
                    item_impedance = ET.SubElement(item_channel, 'impedance')
                    if (channels[i].type.value == ChannelType.UNI.value):
                        item_impedance.text=str(imp_df['impedance'].values[j]) 
                    else:
                        item_impedance.text='N.A.'
                # measurement unit (strongly preferred unit: microvolts)
                item_unit = ET.SubElement(item_channel, 'unit')
                item_unit.text = channels[i].unit_name
                i+=1
                
        #Acquisition meta-data
        de_acquisition = ET.SubElement(de_desc, 'acquisition')
        item_manufacturer =  ET.SubElement(de_acquisition, 'manufacturer')
        item_manufacturer.text = 'TMSi'
        item_model =  ET.SubElement(de_acquisition, 'model')
        item_model.text = 'SAGA'
        item_precision =  ET.SubElement(de_acquisition, 'precision')
        item_precision.text = '32'
        
        #Reference meta-data
        de_reference = ET.SubElement(de_desc, 'reference')
        item_label =  ET.SubElement(de_reference, 'label')
        item_subtracted =  ET.SubElement(de_reference, 'subtracted')
        item_subtracted.text = 'Yes'
        item_common_average =  ET.SubElement(de_reference, 'common_average')
        if self.device.config._reference_method:
            item_label.text = 'average'
            item_common_average.text = 'Yes'
        else:
            item_label.text = 'CREF'
            item_common_average.text = 'No'  
        

        # Write the StreamHeader-cunk
        XdfWriter._write_chunk(self._fp, 4, ChunkTag.stream_header, xml_etree_to_string(de_info))
        
        
    def __write_stream_header_chunk(self, channels, sample_rate, imp_df=None):
        """ Writes the StreamHeader-chunk :
            - <info>-element with:
                - device-name,
                - overall content-type,
                - channel-count,
                - sample-rate and
                - sample-data-format
                - EEG meta-data, the channels-element ('<desc><channels>'):
                    - channel-name (<label>)
                    - channel-type (<type>)
                    - channel-unit-name (<unit>)
                    - channel-location (<location>)
                    - channel-impedance (<impedance>)
                -reference meta-data
                -acquistion meta-data

            Args:
                channels : 'list DeviceChannel' channel list
                sample_rate : int' The rate of the current configuration, with which
                                   sample-sets are sent during a measurement.
                imp_df: 'DataFrame' Previously recorded impedances 
        """    
        # Assemble <info>-root-element:
        num_channels = self._num_channels
        de_info = ET.Element('info')
        item_name = ET.SubElement(de_info, 'name')
        item_name.text = self.device.get_device_type()
        item_type = ET.SubElement(de_info, 'type')
        item_type.text = 'EEG'
        item_channel_count = ET.SubElement(de_info, 'channel_count')
        item_channel_count.text =  str(num_channels)
        item_nominal_srate = ET.SubElement(de_info, 'nominal_srate')
        item_nominal_srate.text =  str(sample_rate)
        item_channel_format = ET.SubElement(de_info, 'channel_format')
        item_channel_format.text = 'float32'

        # Assemble the EEG meta-data
        de_desc = ET.SubElement(de_info, 'desc')
        de_channels = ET.SubElement(de_desc, 'channels')

        #read channel locations
        chLocs=pd.read_csv(join(modules_dir,'TMSiSDK/_resources/EEGchannelsTMSi3D.txt'), sep="\t", header=None)
        chLocs.columns=['default_name', 'eeg_name', 'X', 'Y', 'Z']

        # Meta-data per channel
        for j in range(len(channels)):
            # description of one channel, repeated (one for each channel in the time series)
            item_channel = ET.SubElement(de_channels, 'channel')
            # channel label
            item_label =  ET.SubElement(item_channel, 'label')
            item_label.text = channels[j].get_channel_name()
            # channel content-type (EEG, EMG, EOG, ...)
            item_type = ET.SubElement(item_channel, 'type')
            
            if (channels[j].get_channel_type().value == ChannelType.UNI.value):
                item_type.text = 'EEG'
                #channel location
                if self.add_ch_locs:
                    item_location = ET.SubElement(item_channel, 'location')
                    item_x = ET.SubElement(item_location, 'X')
                    item_y = ET.SubElement(item_location, 'Y')
                    item_z = ET.SubElement(item_location, 'Z')
                    item_x.text=str(95*chLocs['X'].values[j])
                    item_y.text=str(95*chLocs['Y'].values[j])
                    item_z.text=str(95*chLocs['Z'].values[j])
            
            elif (channels[j].get_channel_type().value == ChannelType.BIP.value):
                item_type.text = 'BIP'
            elif (channels[j].get_channel_type().value == ChannelType.AUX.value):
                item_type.text = 'AUX'
            elif (channels[j].get_channel_type().value == ChannelType.sensor.value):
                item_type.text = 'sensor'
            elif (channels[j].get_channel_type().value == ChannelType.status.value):
                item_type.text = 'status'
            elif (channels[j].get_channel_type().value == ChannelType.counter.value):
                item_type.text = 'counter'
            else:
                item_type.text = '-'
                
            # if imp_df is not None:
            #     #channel impedance
            #     item_impedance = ET.SubElement(item_channel, 'impedance')
            #     if (channels[i].type.value == ChannelType.UNI.value):
            #         item_impedance.text=str(imp_df['impedance'].values[j]) 
            #     else:
            #         item_impedance.text='N.A.'
            # measurement unit (strongly preferred unit: microvolts)
            item_unit = ET.SubElement(item_channel, 'unit')
            item_unit.text = channels[j].get_channel_unit_name()
            
                
        #Acquisition meta-data
        de_acquisition = ET.SubElement(de_desc, 'acquisition')
        item_manufacturer =  ET.SubElement(de_acquisition, 'manufacturer')
        item_manufacturer.text = 'TMSi'
        item_model =  ET.SubElement(de_acquisition, 'model')
        item_model.text = self.device.get_device_type()
        item_precision =  ET.SubElement(de_acquisition, 'precision')
        item_precision.text = '32'
        
        #Reference meta-data
        de_reference = ET.SubElement(de_desc, 'reference')
        item_label =  ET.SubElement(de_reference, 'label')
        item_subtracted =  ET.SubElement(de_reference, 'subtracted')
        item_subtracted.text = 'Yes'
        item_common_average =  ET.SubElement(de_reference, 'common_average')
        item_label.text = 'average'
        item_common_average.text = 'Yes'
        
        # Write the StreamHeader-cunk
        XdfWriter._write_chunk(self._fp, 4, ChunkTag.stream_header, xml_etree_to_string(de_info))

    def _write_stream_footer_chunk(self, first_timestamp, last_timestamp, sample_count, sample_rate):
        """ Writes the StreamFooter-chunk :
             - first timestamp in seconds
             - last timestamp in seconds
             - sample-set count
             - sample rate

            Args:
                first_timestamp : timestart measurement start
                last_timestamp : timestart measurement end
                sample_count : Total number of sample-sets written to the xdf-file
                sample_rate : int' The rate of the current configuration, with which
                                   sample-sets are sent during a measurement.

        """
        data = ET.Element('info')
        item_name = ET.SubElement(data, 'first_timestamp')
        item_name.text = str(first_timestamp)
        item_type = ET.SubElement(data, 'last_timestamp')
        item_type.text = str(last_timestamp)
        item_channel_count = ET.SubElement(data, 'sample_count')
        item_channel_count.text =  str(sample_count)
        item_nominal_srate = ET.SubElement(data, 'measured_srate')
        item_nominal_srate.text =  str(sample_rate)

        XdfWriter._write_chunk(self._fp, 4, ChunkTag.stream_footer, xml_etree_to_string(data))


    @staticmethod
    def _write_sample_chunk(f, sample_sets, num_sample_sets, n_chan, pack_struct):
        """ Writes the Samples-chunk :

            Args:
                f : 'file-object' of the xdf-file
                sample_sets : list of <SampleSet>
                num_sample_sets : Number of sample-sets to write

        """
        sample_chunk = bytearray();
        num_sample_bytes = int(4)
        ts = int(0)
        
        sample_chunk += num_sample_bytes.to_bytes(1, 'little')
        sample_chunk += num_sample_sets.to_bytes(4, 'little')
        
        for i in range(num_sample_sets):
            sample_chunk += ts.to_bytes(1, 'little')
            sample_chunk += pack_struct.pack(*sample_sets[i*n_chan:(i+1)*n_chan])
        
        XdfWriter._write_chunk(f, 4, ChunkTag.samples, sample_chunk)

    @staticmethod
    def _write_boundary_chunk(f):
        """ Writes the Boundary-chunk :

            Args:
                f : 'file-object' of the xdf-file
        """
        uuid = [0x43, 0xA5, 0x46, 0xDC, 0xCB, 0xF5, 0x41, 0x0F, 0xB3, 0x0E, 0xD5, 0x46, 0x73, 0x83, 0xCB, 0xE4]
        boundary_chunk = bytearray(uuid)

        XdfWriter._write_chunk(f, 4, ChunkTag.boundary, boundary_chunk)
    
class ConsumerThread(threading.Thread):
    def __init__(self, file_writer, name):
        super(ConsumerThread,self).__init__()
        self.name = name
        self._fw = file_writer
        self.q_sample_sets = file_writer.q_sample_sets
        self.sampling = True
        self._sample_set_block_index = 0
        self._start_time = time.time()
        self._num_sample_sets_per_sample_data_block = file_writer._num_sample_sets_per_sample_data_block
        self._sample_sets_in_block = []
        self._num_written_sample_sets = 0
        self._boundary_chunk_counter_threshold = self._fw._sample_rate * 10
        self._boundary_chunk_counter = 0
        self.pack_struct=file_writer.pack_struct 
        self._remaining_samples=np. array([])

    def run(self):
        
        while ((self.sampling) or (not self.q_sample_sets.empty())) :
            while not self.q_sample_sets.empty():
                #Request sample data
                sd = self.q_sample_sets.get()
                self.q_sample_sets.task_done()
                             
                #Combine with previous data
                if self._remaining_samples.size:
                      samples = np.concatenate((self._remaining_samples,sd.samples))
                else:
                    samples = np.array(sd.samples)
                
                n_samp = int(len(samples) / sd.num_samples_per_sample_set)
                
                try:
                    # Collect the sample-sets:
                    # When collected enough to fill a sample-data-block, write it to a Samples-chunk
                    for i in range(np.int(np.floor(n_samp / self._num_sample_sets_per_sample_data_block))):
                        self._sample_sets_in_block=samples[i*self._num_sample_sets_per_sample_data_block*sd.num_samples_per_sample_set:(i+1)*self._num_sample_sets_per_sample_data_block*sd.num_samples_per_sample_set]
                        XdfWriter._write_sample_chunk(self._fw._fp,\
                                                    self._sample_sets_in_block,\
                                                    self._num_sample_sets_per_sample_data_block, \
                                                    n_chan=sd.num_samples_per_sample_set, \
                                                    pack_struct=self.pack_struct)
                        self._num_written_sample_sets += self._num_sample_sets_per_sample_data_block
                        self._sample_set_block_index += 1
            
              
                        # Write approximately every 10 seconds a Boundary-chunk
                        self._boundary_chunk_counter += self._num_sample_sets_per_sample_data_block
                        if (self._boundary_chunk_counter >= self._boundary_chunk_counter_threshold):
                            XdfWriter._write_boundary_chunk(self._fw._fp)
                            self._boundary_chunk_counter = 0
                    
                    # Store remaining samples for next repetion
                    i = np.int(np.floor(n_samp/self._num_sample_sets_per_sample_data_block))
                    ind = np.arange(i*self._num_sample_sets_per_sample_data_block*sd.num_samples_per_sample_set, n_samp*sd.num_samples_per_sample_set)
                    if ind.any:
                        self._remaining_samples=samples[ind]
                    else:
                        self._remaining_samples=np. array([])

                except:
                    raise TMSiError(TMSiErrorCode.file_writer_error)

            time.sleep(0.01)
       
        #Handle remaining samples before closing file
        if self._remaining_samples.any():
            self._sample_sets_in_block = np.zeros(self._num_sample_sets_per_sample_data_block*sd.num_samples_per_sample_set)
            self._sample_sets_in_block[:len(self._remaining_samples)] = self._remaining_samples
            
            XdfWriter._write_sample_chunk(self._fw._fp,\
                                                    self._sample_sets_in_block,\
                                                    self._num_sample_sets_per_sample_data_block, n_chan=sd.num_samples_per_sample_set, pack_struct=self.pack_struct)

        
        # When done : write the StreamFoot-cunk and close the file
        elapsed_time = time.time() - self._start_time
        self._fw._write_stream_footer_chunk(0, int(elapsed_time), self._num_written_sample_sets, self._fw._sample_rate)

        print(self.name, " ready, closing file")
        self._fw._fp.close()
        return

    def stop_sampling(self):
        print(self.name, " stop sampling")
        self.sampling = False;