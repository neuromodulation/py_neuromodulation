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
 * @file ${settings.py} 
 * @brief Module for SDK 'locally' used settings
 *
 */


'''

def _initialize():
    """
    Instantiates and initializes 'local' static variables fro SDK-modules.
    This method must be called once befor start using the SDK.

    Returns
    -------
    None.

    """
    global _consumer_list # used in <sample_data_server.py> : list of registered
    _consumer_list = []   # consumer for receipt of sample-data
