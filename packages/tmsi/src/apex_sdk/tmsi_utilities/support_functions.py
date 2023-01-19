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
 * @file support_functions.py 
 * @brief 
 * Generic functions needed in the SDK.
 */


'''

import ctypes.wintypes
import os
import struct

def array_to_matrix(input_array, n_channels):
    n_samples = len(input_array) // n_channels
    return [input_array[ii:n_samples * n_channels:n_channels]
            for ii in range(n_channels)]

def matrix_to_multiplexed_array(input_matrix):
    return [input_matrix[i][j] for j in range(len(input_matrix[0])) for i in range(len(input_matrix))] 

def float_to_uint(f):
    fmt_pack='f'*len(f)
    fmt_unpack='I'*len(f)
    pack_struct=struct.Struct(fmt_pack)
    return struct.unpack(fmt_unpack, pack_struct.pack(*list(f)))

def get_documents_path():
    CSIDL_PERSONAL = 5
    SHGFP_TYPE_CURRENT = 0
    buf= ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
    ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_PERSONAL, None, SHGFP_TYPE_CURRENT, buf)
    return buf.value

def get_local_app_data():
    return os.getenv('LOCALAPPDATA')
