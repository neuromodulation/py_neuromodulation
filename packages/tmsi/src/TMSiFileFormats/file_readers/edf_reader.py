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
 * @brief EDF to MNE Converter.
 *
 */


'''

import tkinter as tk
from tkinter import filedialog
import mne
import pandas as pd

from os.path import join, dirname, realpath
Reader_dir = dirname(realpath(__file__)) # directory of this file
modules_dir = join(Reader_dir, '../../') # directory with all modules

class Edf_Reader:
    def __init__(self, filename=None, add_ch_locs=False):
        if filename==None:
            root = tk.Tk()
            filename = filedialog.askopenfilename(title = 'Select edf-file', filetypes = (('edf-files', '*.edf'),('All files', '*.*')))
            root.withdraw()
            
        # read raw edf-file
        # change channel type of COUNTER channel to misc
        mne_object=mne.io.read_raw_edf(filename, misc=[-2], preload=True)
        
        if add_ch_locs:
            # add channel locations from txt file
            chLocs=pd.read_csv(join(modules_dir,'TMSiSDK/_resources/EEGchannelsTMSi3D.txt'), sep="\t", header=None)
            chLocs.columns=['default_name', 'eeg_name', 'X', 'Y', 'Z']
            # add locations and convert to head size of 95 mm
            for idx, ch in enumerate(mne_object.info['chs']):
                try:
                    a=[i for i, e in (enumerate(chLocs['eeg_name'].values) or enumerate(chLocs['default_name'].values)) if e == ch['ch_name']]
                    mne_object.info['chs'][idx]['loc'][0]=95*1e-3*chLocs['X'].values[a]
                    mne_object.info['chs'][idx]['loc'][1]=95*1e-3*chLocs['Y'].values[a]
                    mne_object.info['chs'][idx]['loc'][2]=95*1e-3*chLocs['Z'].values[a]  
                except:
                    pass

        # unit conversion of eeg channels
        mne_object.apply_function(lambda x: x*1e-6, picks='eeg')
        
        self.mne_object=mne_object
        
    def add_impedances(self, imp_filename=None):
        """Add impedances from .txt-file """
        if imp_filename==None:
            root = tk.Tk()
            imp_filename = filedialog.askopenfilename(title = 'Select impedance file', filetypes = (('text files', '*.txt'),('All files', '*.*')))
            root.withdraw()
        impedances = []
        
        imp_df = pd.read_csv(imp_filename, delimiter = "\t", header=None)    
        imp_df.columns=['ch_name', 'impedance', 'unit']
        
        for ch in range(len(self.mne_object.info['chs'])):
            for i_ch in range(len(imp_df)):
                if self.mne_object.info['chs'][ch]['ch_name'] == imp_df['ch_name'][i_ch]:
                    impedances.append(imp_df['impedance'][i_ch])
                    
        self.mne_object.impedances = impedances