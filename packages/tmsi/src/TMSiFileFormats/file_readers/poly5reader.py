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
 * @file ${poly5reader.py} 
 * @brief Poly5 File Reader.
 *
 */


'''

import numpy as np
import struct
import datetime
import mne
import tkinter as tk
from tkinter import filedialog

class Poly5Reader: 
    def __init__(self, filename=None, readAll = True):
        if filename==None:
            root = tk.Tk()

            filename = filedialog.askopenfilename(title = 'Select poly5-file', filetypes = (('poly5-files', '*.poly5'),('All files', '*.*')))
            root.withdraw()
            
        self.filename = filename
        self.readAll = readAll
        print('Reading file ', filename)
        self._readFile(filename)
        
    def read_data_MNE(self,) -> mne.io.RawArray:
        """Return MNE RawArray given internal channel names and types

        Returns
        -------
        mne.io.RawArray
        """

        streams = self.channels
        fs = self.sample_rate
        labels = [s._Channel__name for s in streams]
        units = [s._Channel__unit_name for s in streams]

        type_options = [
            "ecg",
            "bio",
            "stim",
            "eog",
            "misc",
            "seeg",
            "dbs",
            "ecog",
            "mag",
            "eeg",
            "ref_meg",
            "grad",
            "emg",
            "hbr",
            "hbo",
        ]
        types_clean = []
        for t in labels:
            for t_option in type_options:
                if t_option in t.lower():
                    types_clean.append(t_option)
                    break
            else:
                types_clean.append("misc")

        info = mne.create_info(ch_names=labels, sfreq=fs, ch_types=types_clean)

        # convert from microvolts to volts if necessary
        scale = np.array([1e-6 if u == "µVolt" else 1 for u in units])

        raw = mne.io.RawArray(self.samples * np.expand_dims(scale, axis=1), info)
        return raw
        
    def _readFile(self, filename):
        try:
            self.file_obj = open(filename, "rb")
            file_obj = self.file_obj
            try:    
                self._readHeader(file_obj)
                self.channels = self._readSignalDescription(file_obj)
                self._myfmt = 'f' * self.num_channels*self.num_samples_per_block
                self._buffer_size = self.num_channels*self.num_samples_per_block
                
                if self.readAll:
                    sample_buffer = np.zeros(self.num_channels * self.num_samples)
     
                    for i in range(self.num_data_blocks):
                        print('\rProgress: % 0.1f %%' %(100*i/self.num_data_blocks), end="\r")
                         
                        # Check whether final data block is filled completely or not
                        if i == self.num_data_blocks-1:
                            _final_block_size = self.num_samples / self.num_data_blocks
                            if _final_block_size % self.num_samples_per_block != 0:
                                data_block = self._readSignalBlock(file_obj, 
                                                                   buffer_size = (self.num_samples%self.num_samples_per_block) * self.num_channels, 
                                                                   myfmt = 'f' * (self.num_samples%self.num_samples_per_block) * self.num_channels)
                            else:
                                data_block = self._readSignalBlock(file_obj, self._buffer_size, self._myfmt)
                        else:
                            data_block = self._readSignalBlock(file_obj, self._buffer_size, self._myfmt)
                        
                        # Get indices that need to be filled in the samples array
                        i1 = i * self.num_samples_per_block * self.num_channels
                        i2 = (i+1) * self.num_samples_per_block * self.num_channels
                        
                        # Correct for final data block if this is not fully filled
                        if i2 >= self.num_samples * self.num_channels:
                            i2 = self.num_samples * self.num_channels
                        
                        # Insert the read data_block into the sample_buffer array
                        sample_buffer[i1:i2] = data_block
                       
                    samples=np.transpose(np.reshape(sample_buffer, [self.num_samples, self.num_channels]))
                    
                    self.ch_names = [s._Channel__name for s in self.channels]
                    self.ch_unit_names = [s._Channel__unit_name for s in self.channels]
                    
                    self.samples=samples
                    print('Done reading data.')
                    self.file_obj.close()
                    
            except Exception as e:
                print('Reading data failed, because of the following error:\n')
                raise
        except OSError:
            print('Could not open file. ')
        
        
    def readSamples(self, n_blocks = None):
        "Function to read a subset of sample blocks from a file"
        if n_blocks==None:
            n_blocks = self.num_data_blocks
            
        sample_buffer = np.zeros(self.num_channels*n_blocks*self.num_samples_per_block)
     
        for i in range(n_blocks):
            data_block = self._readSignalBlock(self.file_obj, self._buffer_size, self._myfmt)
            i1 = i * self.num_samples_per_block * self.num_channels
            i2 = (i+1) * self.num_samples_per_block * self.num_channels
            sample_buffer[i1:i2] = data_block
        
        samples = np.transpose(np.reshape(sample_buffer, [self.num_samples_per_block*(i+1), self.num_channels]))
        return samples
    
            
    def _readHeader(self, f):
        header_data = struct.unpack("=31sH81phhBHi4xHHHHHHHiHHH64x", f.read(217))
        magic_number = str(header_data[0])
        version_number = header_data[1]
        self.sample_rate = header_data[3]
        # self.storage_rate=header_data[4]
        self.num_channels = header_data[6]//2
        self.num_samples = header_data[7]
        self.start_time = datetime.datetime(header_data[8], header_data[9], 
                                            header_data[10], header_data[12], 
                                            header_data[13], header_data[14])
        self.num_data_blocks = header_data[15]
        self.num_samples_per_block = header_data[16]
        if magic_number !="b'POLY SAMPLE FILEversion 2.03\\r\\n\\x1a'":
            print('This is not a Poly5 file.')
        elif  version_number != 203:
            print('Version number of file is invalid.')
        else:
            print('\t Number of samples:  %s ' %self.num_samples)
            print('\t Number of channels:  %s ' % self.num_channels)
            print('\t Sample rate: %s Hz' %self.sample_rate)
            
            
    def _readSignalDescription(self, f): 
        chan_list = []
        for ch in range(self.num_channels):
            channel_description = struct.unpack("=41p4x11pffffH62x", f.read(136))
            name = channel_description[0][5:].decode('ascii')
            unit_name = channel_description[1].decode('utf-8')
            ch = Channel(name, unit_name)
            chan_list.append(ch)
            f.read(136)
        return chan_list
        
            
    
    def _readSignalBlock(self, f, buffer_size, myfmt):
        f.read(86)
        sampleData = f.read(buffer_size*4)
        DataBlock = struct.unpack(myfmt, sampleData)
        SignalBlock = np.asarray(DataBlock)
        return SignalBlock
    
    def close(self):
        self.file_obj.close()
        

class Channel:
    """ 'Channel' represents a device channel. It has the next properties:

        name : 'string' The name of the channel.

        unit_name : 'string' The name of the unit (e.g. 'μVolt)  of the sample-data of the channel.
    """

    def __init__(self, name, unit_name):
        self.__unit_name = unit_name
        self.__name = name
        
        
        
if __name__ == "__main__":
    data = Poly5Reader()
