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
 * @file dongle_info.py 
 * @brief 
 * APEX Dongle information.
 */


'''

from .apex_const import ApexConst

class DongleInfo():
    def __init__(self):
        self.TMSiDongleID = ApexConst.TMSI_DONGLE_ID_NONE
        self.SerialNumber = ApexConst.TMSI_DONGLE_ID_NONE
