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
 * @file ${xdf_reader.py} 
 * @brief XDF File Reader.
 *
 */


'''

from pyxdf import load_xdf
import mne
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import copy

from os.path import join, dirname, realpath
Reader_dir = dirname(realpath(__file__)) # directory of this file
modules_dir = join(Reader_dir, '../../') # directory with all modules


class Xdf_Reader: 
    def __init__(self, filename=None, add_ch_locs=False):
        if filename==None:
            root = tk.Tk()

            filename = filedialog.askopenfilename(title = 'Select xdf-file', filetypes = (('xdf-files', '*.xdf'),('All files', '*.*')))
            root.withdraw()
            
        self.filename = filename
        self.add_ch_locs=add_ch_locs
        print('Reading file ', filename)
        self.data, self.time_stamps = self._readFile(filename)
        
    def _readFile(self, fname):
        try: 
            streams, header = load_xdf(fname)
            num_streams = len(streams)
            self.stream_info = {}
            
            print('Number of streams in file: ' + str(num_streams))
            for i in range(num_streams):
                stream=streams[i]
                self.stream_info[i] = stream["info"]
                if stream is not None:
                  fs = float(stream["info"]["nominal_srate"][0])
              
                  labels, types, units, impedances = self._get_ch_info(stream)
                  
                  type_options=["ecg", "bio", "stim", "eog", "misc", "seeg", "dbs", "ecog", "mag", "eeg", "ref_meg", "grad", "emg", "hbr", "hbo"]
                  for ind, t in enumerate(types):
                      if t=="EEG":
                          types[ind]="eeg"
                      elif not t in type_options:
                          types[ind]="misc"
                  info = mne.create_info(ch_names=labels, sfreq=fs, ch_types=types)   
                  info=self._get_ch_locations(stream, info)
                  if self.add_ch_locs:
                      info=self._add_ch_locations(info)
                 
                  # convert from microvolts to volts if necessary
                  scale = np.array([1e-6 if u == "µVolt" else 1 for u in units])
                  raw = mne.io.RawArray((stream["time_series"] * scale).T, info)
                  raw.impedances=impedances
                  
                  
                  if raw is not None:
                      print(raw, end="\n\n")
                      print(raw.info)
                      
                      if num_streams == 1:
                          return (raw,), (stream['time_stamps'],)
                      else:
                          if i == 0:
                              output_data = (copy.copy(raw),)
                              output_timestamps = (copy.copy(stream['time_stamps']),)
                          elif i == num_streams - 1:
                              output_data = output_data + (copy.copy(raw),)
                              output_timestamps = output_timestamps + (copy.copy(stream['time_stamps']),)
                              return output_data, output_timestamps
                          else:
                              output_data = output_data + (copy.copy(raw),)
                              output_timestamps = output_timestamps + (copy.copy(stream['time_stamps']),)
        except Exception as e:
            print('Reading data failed because of the following error:\n')
            raise
            
    def add_impedances(self, imp_filename=None):
        """Add impedances from .txt-file """
        if imp_filename==None:
            root = tk.Tk()
            imp_filename = filedialog.askopenfilename(title = 'Select impedance file', filetypes = (('text files', '*.txt'),('All files', '*.*')))
            root.withdraw()
        impedances = []
        
        imp_df = pd.read_csv(imp_filename, delimiter = "\t", header=None)    
        imp_df.columns=['ch_name', 'impedance', 'unit']
        
        for ch in range(len(self.data[0].info['chs'])):
            for i_ch in range(len(imp_df)):
                if self.data[0].info['chs'][ch]['ch_name'] == imp_df['ch_name'][i_ch]:
                    impedances.append(imp_df['impedance'][i_ch])
                    
        self.data[0].impedances = impedances
   
    def _get_ch_info(self, stream):
        # read channel labels, types, units and impedances
        labels, types, units, impedances = [], [], [], []

        for ch in stream["info"]["desc"][0]["channels"][0]["channel"]:
            labels.append(str(ch["label"][0]))
            types.append(ch["type"][0])
            units.append(ch["unit"][0])
            if ch["impedance"]:
                impedances.append(str(ch["impedance"][0]))
        return labels, types, units, impedances
    
    def _get_ch_locations(self, stream, info):
        # read channel locations and convert unit from mm to m
        for i, ch in enumerate(stream["info"]["desc"][0]["channels"][0]["channel"]):
            if ch["location"]:
                info['chs'][i]['loc'][0]=float(ch["location"][0]["X"][0])*1e-3
                info['chs'][i]['loc'][1]=float(ch["location"][0]["Y"][0])*1e-3
                info['chs'][i]['loc'][2]=float(ch["location"][0]["Z"][0])*1e-3
                self.add_ch_locs=False
        return info
    
    def _add_ch_locations(self, info):
        # add channel locations from txt file
        chLocs=pd.read_csv(join(modules_dir,'TMSiSDK/_resources/EEGchannelsTMSi3D.txt'), sep="\t", header=None)
        chLocs.columns=['default_name', 'eeg_name', 'X', 'Y', 'Z']
        
        # add locations and convert to head size of 95 mm
        for idx, ch in enumerate(info['chs']):
            try:
                a=[i for i, e in (enumerate(chLocs['eeg_name'].values) or enumerate(chLocs['default_name'].values)) if e == ch['ch_name']]
                info['chs'][idx]['loc'][0]=95*1e-3*chLocs['X'].values[a]
                info['chs'][idx]['loc'][1]=95*1e-3*chLocs['Y'].values[a]
                info['chs'][idx]['loc'][2]=95*1e-3*chLocs['Z'].values[a]  
            except:
                pass
        return info
        
    def get_stream_info(self):
        # Retrieve the information from the streams in the data. 
        if hasattr(self, "stream_info"):
            return self.stream_info
        else:
            return None