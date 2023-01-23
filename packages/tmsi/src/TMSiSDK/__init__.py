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
 * @file ${__init__.py} 
 * @brief Initialisation of the TMSiSDK directory classes.
 *
 */


"""
from os.path import join, dirname, realpath

from . import (
    device,
    error,
    sample_data_server,
    sample_data,
    settings,
    tmsi_device,
)
from . import devices
from .device import DeviceInterfaceType, DeviceState
from .devices.saga import SagaDevice
from .devices.saga import xml_saga_config
from .error import TMSiError, TMSiErrorCode, DeviceErrorLookupTable
from .tmsi_device import DeviceType, discover, initialize


def get_config(config_name):
    TMSiSDK_dir = dirname(realpath(__file__))  # directory of this file
    configs_dir = join(TMSiSDK_dir, "configs")
    config_fname = join(configs_dir, config_name + ".xml")
    return config_fname
