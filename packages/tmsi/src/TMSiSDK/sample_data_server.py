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
 * @file ${sample_data_server.py} 
 * @brief Sample Data Server module
 *
 */


"""
import queue
from TMSiSDK import settings
from copy import copy


class SampleDataConsumer:
    """Local class which identifies which consumer registered for what device"""

    def __init__(self, id, q):
        self.id = id
        self.q = q


def registerConsumer(id, q):
    """Registers a consumer-queue to receive the sample-data of a specific
    device.
    This method is called by sample-data-consumers.

    Args:
        id: <int> : Unique id of a device <Device.id>.
            Indicates from which specific device, received sample-data
            must be put into the registered <queue>

        q: <queue> The queue into which received sample-data will be put.
    """
    settings._consumer_list.append(SampleDataConsumer(id, q))


def unregisterConsumer(id, q):
    """Unregisters a consumer-queue associated with a file-writer or plotter
    object.
    This method is called by close-methods of the sample-data-consumers.

    Args:
        id: <int> : Unique id of a device <Device.id>.
            Indicates from which specific device, received sample-data
            must be put into the registered <queue>

        q: <queue> The queue object which has to be removed from the global
            consumer list.
    """
    num_consumers = len(settings._consumer_list)
    for i in range(num_consumers):
        if settings._consumer_list[i].id == id:
            if settings._consumer_list[i].q == q:
                idx_remove = copy(i)
    # try:
    #     settings._consumer_list[idx_remove].q.close()
    # except AttributeError:
    #     pass
    settings._consumer_list.pop(idx_remove)


def putSampleData(id, data):
    """Puts a <SampleData>-object, coming from device <Device.id> into the queues
    of registered consumers.
    This method is called by sample-data-producers.

    Args:
        id: <int> : Unique id of the device (<Device>.id), which forwards
            received sample-data to the sample-data-server.

        data <SampleData> The sample-data.
    """
    num_consumers = len(settings._consumer_list)
    for i in range(num_consumers):
        if settings._consumer_list[i].id == id:
            try:
                settings._consumer_list[i].q.put(data, timeout=5)
            except TypeError:
                settings._consumer_list[i].q.put(data)
            except:
                print(f"Failed to put raw data. Queue full: ", id)
