"""
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
 * @file ${poly5_to_edf_converter.py} 
 * @brief Poly5 to EDF converter
 *
 */


"""

import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

from os.path import join, dirname, realpath

Reader_dir = dirname(realpath(__file__))  # directory of this file
modules_dir = join(Reader_dir, "../../")  # directory with all modules

from TMSiFileFormats.file_readers import Poly5Reader
from scipy.signal import sosfiltfilt, butter
from EDFlib.edfwriter import EDFwriter


class Poly5_to_EDF_Converter:
    def __init__(self, batch=None, filename=None, foldername=None, f_c=0.1):
        """Converts poly5-file(s) to edf format. Either a single file or all
        poly5 files in a folder and its subfolders.
        batch: True or False: convert batch of files or single file
        filename: full path to file
        foldername: full path to folder
        f_c: single value or list of two values; cut-off frequency/frequencies of high pass or bandpass filter"""
        self.f_c = f_c
        if not batch:
            if filename == None:
                root = tk.Tk()
                filename = filedialog.askopenfilename()
                root.withdraw()
            self.convertFile(filename)
        else:
            if foldername == None:
                root = tk.Tk()
                foldername = filedialog.askdirectory()
                root.withdraw()
            n_poly5 = 0
            conversion_files = []
            for root, dirs, files in os.walk(foldername):
                for file in files:
                    if file.endswith(".poly5") or file.endswith(".Poly5"):
                        n_poly5 = n_poly5 + 1
                        edf_filename = file.replace("poly5", "edf")
                        edf_filename = edf_filename.replace("Poly5", "edf")
                        if not os.path.isfile(os.path.join(root, edf_filename)):
                            conversion_files.append(os.path.join(root, file))
            print("Convert all poly5-files in folder and subfolders to edf")
            print("\tFolder: ", foldername)
            print("\tTotal number of poly5-files: ", n_poly5)
            print(
                "\tFiles already converted: ", n_poly5 - len(conversion_files)
            )
            print("\tFiles to be converted: ", len(conversion_files))
            for i in range(len(conversion_files)):
                print("\n\nConvert file ", i + 1, "of", len(conversion_files))
                self.convertFile(conversion_files[i])

    def convertFile(self, filename):
        self._readData(filename)
        self._remove_empty_samples()
        self._filter_data()
        self._write_edf_meta_data()
        self._write_edf_data()

    def _readData(self, filename):
        self.data = Poly5Reader(filename)
        self.fs = self.data.sample_rate
        self.n_signals = len(self.data.samples)

        self.n_analogue = 0
        for chan in range(0, self.n_signals):
            if "Volt" in self.data.ch_unit_names[chan]:
                # analogue channels
                self.n_analogue = self.n_analogue + 1

    def _remove_empty_samples(self):
        """remove padding zeros, based on COUNTER channel"""
        empty_samples = list(np.where(self.data.samples[-1, :] == 0)[0])
        self.data.samples = np.delete(self.data.samples, empty_samples, axis=1)

    def _filter_data(self):
        """low-pass filter data of analogue channels to remove offset and drift"""
        if not isinstance(self.f_c, list):
            print(
                "Data is low-pass filtered with cut-off frequency ",
                self.f_c,
                "Hz",
            )
            sos = butter(
                1, self.f_c / (self.fs / 2), btype="highpass", output="sos"
            )
            self.data.samples[: self.n_analogue, :] = sosfiltfilt(
                sos, self.data.samples[: self.n_analogue, :]
            )
        else:
            print(
                "Data is band-pass filtered with cut-off frequencies ",
                self.f_c[0],
                "Hz and",
                self.f_c[1],
                "Hz",
            )
            sos = butter(
                1,
                [self.f_c[0] / (self.fs / 2), self.f_c[1] / (self.fs / 2)],
                btype="bandpass",
                output="sos",
            )
            self.data.samples[: self.n_analogue, :] = sosfiltfilt(
                sos, self.data.samples[: self.n_analogue, :]
            )

    def _write_edf_meta_data(self):
        """ "write edf meta-data to file"""
        self.edf_filename = self.data.filename.replace("poly5", "edf")
        self.edf_filename = self.edf_filename.replace("Poly5", "edf")
        print("Writing to file ", self.edf_filename)

        self.hdl = EDFwriter(
            self.edf_filename, EDFwriter.EDFLIB_FILETYPE_EDFPLUS, self.n_signals
        )

        for chan in range(0, self.n_signals):
            # write sample frequency, channel name and dimension
            self.hdl.setSampleFrequency(chan, self.fs)
            self.hdl.setSignalLabel(chan, self.data.ch_names[chan])
            self.hdl.setPhysicalDimension(chan, self.data.ch_unit_names[chan])

            # write minima and maxima
            if max(self.data.samples[chan, :]) == min(
                self.data.samples[chan, :]
            ):
                self.hdl.setPhysicalMaximum(
                    chan, max(self.data.samples[chan, :]) + 100
                )
                self.hdl.setPhysicalMinimum(
                    chan, min(self.data.samples[chan, :])
                )
                self.hdl.setDigitalMaximum(chan, 32767)
                self.hdl.setDigitalMinimum(chan, -32768)
            elif "V" in self.data.ch_unit_names[chan]:
                # analogue channels
                self.hdl.setPhysicalMaximum(
                    chan, max(self.data.samples[chan, :])
                )
                self.hdl.setPhysicalMinimum(
                    chan, min(self.data.samples[chan, :])
                )
                self.hdl.setDigitalMaximum(chan, 32767)
                self.hdl.setDigitalMinimum(chan, -32768)
                if not isinstance(self.f_c, list):
                    self.hdl.setPreFilter(chan, "HP:" + str(self.f_c) + "Hz")
                else:
                    self.hdl.setPreFilter(
                        chan,
                        "HP:"
                        + str(self.f_c[0])
                        + "Hz LP:"
                        + str(self.f_c[1])
                        + "Hz",
                    )
            elif "count" in self.data.ch_names[chan].lower():
                # counter channel
                self.hdl.setPhysicalMaximum(chan, self.fs)
                self.hdl.setPhysicalMinimum(chan, 0)
                self.hdl.setDigitalMaximum(chan, self.fs)
                self.hdl.setDigitalMinimum(chan, 0)
            else:
                # other digital channels
                self.hdl.setPhysicalMaximum(
                    chan, max(self.data.samples[chan, :])
                )
                if min(self.data.samples[chan, :]) < 0:
                    self.hdl.setPhysicalMinimum(
                        chan, min(self.data.samples[chan, :])
                    )
                else:
                    self.hdl.setPhysicalMinimum(chan, 0)
                self.hdl.setDigitalMaximum(chan, 32767)
                self.hdl.setDigitalMinimum(chan, -32768)

    def _write_edf_data(self):
        """write data to edf-file"""
        n_blocks = np.int64(
            np.floor((np.size(self.data.samples) / self.n_signals) / self.fs)
        )
        fs = self.fs

        for i in range(0, n_blocks):
            for j in range(0, self.n_signals - 1):

                self.hdl.writeSamples(
                    self.data.samples[j, i * self.fs : (i + 1) * fs]
                )

            j = j + 1
            self.hdl.writeSamples(
                self.data.samples[j, i * fs : (i + 1) * fs] % fs
            )

            print("\rProgress: % 0.1f %%" % (100 * i / n_blocks), end="\r")

        self.hdl.close()
        print("Done writing data")
