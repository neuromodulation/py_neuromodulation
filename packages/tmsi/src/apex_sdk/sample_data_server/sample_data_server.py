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
 * @file sample_data_server.py 
 * @brief 
 * Singleton class which handles the acquisition of data from the device and make them available to consumers.
 */


'''

from queue import Queue
from copy import copy

from ..sample_data_server.sample_data import SampleDataConsumer, SampleData
from ..sample_data_server.event_data import EventDataConsumer, EventData
from ..tmsi_utilities.singleton import Singleton


class SampleDataServer(metaclass = Singleton):
    def __init__(self):
        self.__consumer_list = []
        self.__event_consumer_list = []
    
    def get_consumer_list(self) -> list[SampleDataConsumer]:
        """Gets the list of available consumer.

        :return: list of available consumers.
        :rtype: list[SampleDataConsumer]
        """
        return self.__consumer_list

    def get_event_consumer_list(self) -> list[EventDataConsumer]:
        """Gets the list of available event consumer.

        :return: list of available event consumers.
        :rtype: list[EventDataConsumer]
        """
        return self.__event_consumer_list

    def put_event_data(self, id: int, data: EventData):
        """Puts event in the corresponding event consumer.

        :param id: id of the provider the event consumer is reading.
        :type id: int
        :param data: event to deliver to the event consumer.
        :type data: EventData
        """
        num_consumers = len(self.__event_consumer_list)
        for i in range(num_consumers):
            if (self.__event_consumer_list[i].id == id):
                self.__event_consumer_list[i].q.put(data)
    
    def put_sample_data(self, id: int, data: SampleData):
        """Puts data in the corresponding consumer.

        :param id: id of the provider the consumer is reading.
        :type id: int
        :param data: data to deliver to the consumer.
        :type data: SampleData
        """
        num_consumers = len(self.__consumer_list)
        for i in range(num_consumers):
            if (self.__consumer_list[i].id == id):
                self.__consumer_list[i].q.put(data)

    def register_consumer(self, id: int, q: Queue):
        """Creates the new consumer and registers it to the list of consumers.

        :param id: id of the provider.
        :type id: int
        :param q: queue of the consumer.
        :type q: Queue
        """
        self.__consumer_list.append(SampleDataConsumer(id, q))

    def register_event_consumer(self, id: int, q: Queue):
        """Creates the new event consumer and registers it to the list of event consumers.

        :param id: id of the provider.
        :type id: int
        :param q: queue of the event consumer.
        :type q: Queue
        """
        self.__event_consumer_list.append(EventDataConsumer(id, q))

    def unregister_consumer(self, id: int, q: Queue):
        """Unregister the queue from the list of consumers.

        :param id: if of the provider.
        :type id: int
        :param q: queue of the consumer.
        :type q: Queue
        """
        num_consumers = len(self.__consumer_list)
        for i in range(num_consumers):
            if self.__consumer_list[i].id == id: 
                if self.__consumer_list[i].q == q:
                    idx_remove = copy(i)
        self.__consumer_list.pop(idx_remove)

    def unregister_event_consumer(self, id: int, q: Queue):
        """Unregister the queue from the list of event consumers.

        :param id: if of the provider.
        :type id: int
        :param q: queue of the event consumer.
        :type q: Queue
        """
        num_consumers = len(self.__event_consumer_list)
        for i in range(num_consumers):
            if self.__event_consumer_list[i].id == id: 
                if self.__event_consumer_list[i].q == q:
                    idx_remove = copy(i)
        self.__event_consumer_list.pop(idx_remove)

    