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
 * @file ${file_writer.py} 
 * @brief File Writer Interface
 *
 */


"""

from enum import Enum
from TMSiSDK.error import TMSiError, TMSiErrorCode


class FileFormat(Enum):
    none = 0
    poly5 = 1
    xdf = 2
    lsl = 3


class FileWriter:
    """<FileWriter> implements a file-writer for writing sample-data, captured
    during a measurement, to a specific file in a specific data-format.

    Args:
        data_format_type : <FileFormat> Specifies the data-format of the file.
        This can be poly5, gdf+ or gdf.

        filename : <string> The path and name of the file, into which the
        measurement-data must be written.
    """

    def __init__(
        self, data_format_type, filename, add_ch_locs=False, download=False
    ):
        if data_format_type == FileFormat.poly5:
            from .file_formats.poly5_file_writer import Poly5Writer

            self._data_format_type = data_format_type
            self._file_writer = Poly5Writer(filename, download)
        elif data_format_type == FileFormat.xdf:
            from .file_formats.xdf_file_writer import XdfWriter

            self._data_format_type = data_format_type
            self._file_writer = XdfWriter(filename, add_ch_locs)
        elif data_format_type == FileFormat.lsl:
            from .file_formats.lsl_stream_writer import LSLWriter

            self._data_format_type = data_format_type
            self._file_writer = LSLWriter(filename)
        else:
            print("Unsupported data format")
            raise TMSiError(TMSiErrorCode.api_incorrect_argument)

    def open(self, device):
        """Opens a file-writer session.

        Must be called BEFORE a measurement is started.

        Links a 'Device' to a 'Filewriter'-instance and prepares for the
        measurement that will start.

        In the open-method the file-writer-object will prepare for the receipt
        of sample-data from the 'Device' as next:
            - Retrieve the configuration (e.g. channel-list, sample_rate) from the 'Device'
              to write to the file and to prepare for the receipt of sample-data.
            - Register at the 'sample_data_server' for receipt of the sample-data.
            - Create a dedicated sampling-thread, which is responsible to
              processes during the measurement the incoming sample-data.
        """
        self._file_writer.open(device)

    def close(self):
        """Closes an ongoing file-writer session.

        Must be called AFTER a measurement is stopped.

        """
        self._file_writer.close()
