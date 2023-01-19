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
 * @file ${poly5_file_writer.py} 
 * @brief Poly5 File Writer
 *
 */


'''

from datetime import datetime

import os
import threading
import queue
import struct
import time
import numpy as np

from TMSiSDK.error import TMSiError, TMSiErrorCode
from TMSiSDK import sample_data_server
from apex_sdk.device.tmsi_device import TMSiDevice
from apex_sdk.sample_data_server.sample_data_server import SampleDataServer as ApexSampleDataServer 

_QUEUE_SIZE = 1000

class Poly5Writer:
    def __init__(self, filename, download = False):
        self.q_sample_sets = queue.Queue(_QUEUE_SIZE)
        self.device = None
        
        fileparts = filename.split('.')
        if not download:
            now = datetime.now()
            filetime = now.strftime("%Y%m%d_%H%M%S")
            if fileparts[-1]=='poly5' or fileparts[-1]=='Poly5':
                self.filename='.'.join(fileparts[:-1])+ '-' + filetime + '.poly5'
            else:
                self.filename = filename + '-' + filetime + '.poly5'
        else:
            if fileparts[-1]=='poly5' or fileparts[-1]=='Poly5':
                self.filename = filename
            else:
                self.filename = filename + '.poly5'
            
        self._fp = None
        self._date = None

    def open(self, device):
        if isinstance(device, TMSiDevice):
            self.__open_TMSiDevice(device)
            return

        self.device = device
        try:
            self._fp = open(self.filename, 'wb')
            self._date = datetime.now()
            self._sample_rate = device.config.sample_rate
            self._num_channels = len(device.channels)

            # Calculate nr of sample-sets within one sample-data-block:
            # This is the nr of sample-sets in 150 milli-seconds or when the
            # sample-data-block-size exceeds 64kb the it will become the nr of
            # sample-sets that fit in 64kb
            self._num_sample_sets_per_sample_data_block = int(self._sample_rate * 0.15)
            size_one_sample_set = len(self.device.channels) * 4
            if ((self._num_sample_sets_per_sample_data_block * size_one_sample_set) > 64000):
                self._num_sample_sets_per_sample_data_block = int(64000 / size_one_sample_set)

            # Write poly5-header for thsi measurement
            Poly5Writer._writeHeader(self._fp, \
                                     "measurement", \
                                     device.config.sample_rate,\
                                     len(device.channels),\
                                     len(device.channels),\
                                     0,
                                     0,
                                     self._date)
            for (i, channel) in enumerate(self.device.channels):
                Poly5Writer._writeSignalDescription(self._fp, i, channel.name, channel.unit_name)
                
            fmt = 'f'*self._num_channels*self._num_sample_sets_per_sample_data_block 
            self.pack_struct = struct.Struct(fmt)

            sample_data_server.registerConsumer(self.device.id, self.q_sample_sets)

            self._sampling_thread = ConsumerThread(self, name='poly5-writer : dev-id-' + str(self.device.id))
            self._sampling_thread.start()
        except OSError as e:
            print(e)
            raise TMSiError(TMSiErrorCode.file_writer_error)
        except:
            raise TMSiError(TMSiErrorCode.file_writer_error)

    def __open_TMSiDevice(self, device):
        self.device = device
        try:
            self._fp = open(self.filename, 'wb')
            self._date = datetime.now()
            self._sample_rate = self.device.get_device_sampling_frequency()
            self._num_channels = self.device.get_num_channels()

            # Calculate nr of sample-sets within one sample-data-block:
            # This is the nr of sample-sets in 150 milli-seconds or when the
            # sample-data-block-size exceeds 64kb the it will become the nr of
            # sample-sets that fit in 64kb
            self._num_sample_sets_per_sample_data_block = int(self._sample_rate * 0.15)
            size_one_sample_set = self._num_channels * 4
            if ((self._num_sample_sets_per_sample_data_block * size_one_sample_set) > 64000):
                self._num_sample_sets_per_sample_data_block = int(64000 / size_one_sample_set)

            # Write poly5-header for thsi measurement
            Poly5Writer._writeHeader(self._fp, \
                                        "measurement", \
                                        self._sample_rate,\
                                        self._num_channels,\
                                        self._num_channels,\
                                        0,
                                        0,
                                        self._date)
            for (i, channel) in enumerate(self.device.get_device_channels()):
                Poly5Writer._writeSignalDescription(self._fp, i, channel.get_channel_name(), channel.get_channel_unit_name())
                
            fmt = 'f'*self._num_channels*self._num_sample_sets_per_sample_data_block 
            self.pack_struct = struct.Struct(fmt)

            ApexSampleDataServer().register_consumer(self.device.get_id(), self.q_sample_sets)

            self._sampling_thread = ConsumerThread(self, name='poly5-writer : dev-id-' + str(self.device.get_id()))
            self._sampling_thread.start()
        except OSError as e:
            print(e)
            raise TMSiError(TMSiErrorCode.file_writer_error)
        except:
            raise TMSiError(TMSiErrorCode.file_writer_error)

    def close(self):
        # print("Poly5Writer-close")
        self._sampling_thread.stop_sampling()
        
        if isinstance(self.device, TMSiDevice):
            ApexSampleDataServer().unregister_consumer(self.device.get_id(), self.q_sample_sets)
        else:
            sample_data_server.unregisterConsumer(self.device.id, self.q_sample_sets)

    ## Write header of a poly5 file.
    #
    # This function writes the header of a poly5 file to a file.
    # @param f File object
    # @param name Name of measurement
    # @param numSignals Number of signals
    # @param numSamples Number of samples
    # @param numDataBlocks Number of data blocks
    # @param date Date of measurement
    @staticmethod
    def _writeHeader(f, name, sample_rate, num_signals, num_samples, num_data_blocks, num_sample_sets_per_sample_data_block, date):

        data = struct.pack("=31sH81phhBHi4xHHHHHHHiHHH64x",
            b"POLY SAMPLE FILEversion 2.03\r\n\x1a",
            203,
            bytes(name, 'ascii'),
            int(sample_rate),
            int(sample_rate),
            0,
            num_signals * 2,
            num_samples,
            date.year,
            date.month,
            date.day,
            date.isoweekday() % 7,
            date.hour,
            date.minute,
            date.second,
            num_data_blocks,
            num_sample_sets_per_sample_data_block,
            num_signals * 2 * num_sample_sets_per_sample_data_block * 2,
            0
        )
        f.write(data)

    ## Write a signal description
    #
    # @param f File object
    # @param index Index of the signal description
    # @param name Name of the signal (channel)
    # @param unitname The unit name of the signal
    @staticmethod
    def _writeSignalDescription(f, index, name, unit_name):
        data = struct.pack("=41p4x11pffffH62x",
            bytes("(Lo) " + name, 'ascii'),
            bytes(unit_name, 'utf-8'),
            0.0, 1000.0, 0.0, 1000.0,
            index
        )
        f.write(data)

        data = struct.pack("=41p4x11pffffH62x",
            bytes("(Hi) " + name, 'ascii'),
            bytes(unit_name, 'utf-8'),
            0.0, 1000.0, 0.0, 1000.0,
            index
        )
        f.write(data)

    ## Write a signal block
    #
    # @param f File object
    # @param index Index of the data block
    # @param date Date of the sample_data block (measurement)
    # @param signals A list of sample_data, containing NumPy arrays
    @staticmethod
    def _writeSignalBlock(f, index, date, sample_sets_block, num_sample_sets_per_sample_data_block, n_chan, pack_struct):
        data = struct.pack("=i4xHHHHHHH64x",
            int(index * num_sample_sets_per_sample_data_block),
            date.year,
            date.month,
            date.day,
            date.isoweekday() % 7,
            date.hour,
            date.minute,
            date.second
        )
        f.write(data)
        
        sample_sets_block[n_chan-1::n_chan] = sample_sets_block[n_chan-1::n_chan] % (2**24)
        
        bin = pack_struct.pack(*sample_sets_block)
        f.write(bin)

class ConsumerThread(threading.Thread):
    def __init__(self, file_writer, name):
        super(ConsumerThread,self).__init__()
        self.name = name
        self.q_sample_sets = file_writer.q_sample_sets
        self.sampling = True;
        self._sample_set_block_index = 0;
        self._date = file_writer._date
        self._fp = file_writer._fp
        self._sample_rate = file_writer._sample_rate
        self._num_channels = file_writer._num_channels
        self._num_sample_sets_per_sample_data_block = file_writer._num_sample_sets_per_sample_data_block
        self._sample_sets_in_block = []
        self.pack_struct = file_writer.pack_struct 
        self._remaining_samples = np.array([])

    def run(self):
        # print(self.name, " started")   
        
        while ((self.sampling) or (not self.q_sample_sets.empty())) :
            while not self.q_sample_sets.empty():
                sd = self.q_sample_sets.get()
                self.q_sample_sets.task_done()
                
                if self._remaining_samples.size:
                     samples = np.concatenate((self._remaining_samples,sd.samples))
                else:
                    samples = np.array(sd.samples)
                
                n_samp = int(len(samples) / sd.num_samples_per_sample_set)
               
                
                try:
                    for i in range(np.int(np.floor(n_samp/self._num_sample_sets_per_sample_data_block))):
                        self._sample_sets_in_block = samples[i*self._num_sample_sets_per_sample_data_block*sd.num_samples_per_sample_set : (i+1)*self._num_sample_sets_per_sample_data_block*sd.num_samples_per_sample_set]
                        Poly5Writer._writeSignalBlock(self._fp,\
                                                        self._sample_set_block_index,\
                                                        self._date,\
                                                        self._sample_sets_in_block,\
                                                        self._num_sample_sets_per_sample_data_block,\
                                                        self._num_channels, \
                                                        self.pack_struct)
                        self._sample_set_block_index += 1
                    
                        if not (self._sample_set_block_index % 20): 
                            # Go back to start and rewrite header
                            self._fp.seek(0)
                            Poly5Writer._writeHeader(self._fp,\
                                                      "measurement",\
                                                      self._sample_rate,\
                                                      self._num_channels,\
                                                      self._sample_set_block_index * self._num_sample_sets_per_sample_data_block,\
                                                      self._sample_set_block_index,\
                                                      self._num_sample_sets_per_sample_data_block,\
                                                      self._date)
        
                            # Flush all data from buffers to the file
                            self._fp.flush()
                            os.fsync(self._fp.fileno())
        
                            # Go back to end of file
                            self._fp.seek(0, os.SEEK_END)
                        
                    i = np.int(np.floor(n_samp / self._num_sample_sets_per_sample_data_block))
                    ind = np.arange(i*self._num_sample_sets_per_sample_data_block*sd.num_samples_per_sample_set, n_samp*sd.num_samples_per_sample_set)
                    if ind.any:
                        self._remaining_samples = samples[ind]
                    else:
                        self._remaining_samples = np.array([])

                except:
                    raise TMSiError(TMSiErrorCode.file_writer_error)

            time.sleep(0.01)

        while self._remaining_samples.any():
            # Last data block is omitted from saving, to prevent an incomplete data block being part of the Poly5 file
            # This would result in 0s at the end of the file
            if np.shape(self._remaining_samples)[0] < self._num_sample_sets_per_sample_data_block*self._num_channels:
                self._remaining_samples = np.array([])

            else:
                self._sample_sets_in_block = self._remaining_samples[:self._num_sample_sets_per_sample_data_block*self._num_channels]
                
                self._remaining_samples = self._remaining_samples[self._num_sample_sets_per_sample_data_block*self._num_channels:]
                
                Poly5Writer._writeSignalBlock(self._fp,\
                    self._sample_set_block_index,\
                    self._date,\
                    self._sample_sets_in_block,\
                    self._num_sample_sets_per_sample_data_block,\
                    self._num_channels, \
                    self.pack_struct)
                self._sample_set_block_index += 1
        
        # Go back to start and rewrite header
        self._fp.seek(0)
        Poly5Writer._writeHeader(self._fp,\
                                  "measurement",\
                                  self._sample_rate,\
                                  self._num_channels,\
                                  self._sample_set_block_index * self._num_sample_sets_per_sample_data_block,\
                                  self._sample_set_block_index,\
                                  self._num_sample_sets_per_sample_data_block,\
                                  self._date)

        # Flush all data from buffers to the file
        self._fp.flush()
        os.fsync(self._fp.fileno())

        # Go back to end of file
        self._fp.seek(0, os.SEEK_END)
        
        # print(self.name, " ready, closing file")
        self._fp.close()
        return

    def stop_sampling(self):
        # print(self.name, " stop sampling")
        self.sampling = False;