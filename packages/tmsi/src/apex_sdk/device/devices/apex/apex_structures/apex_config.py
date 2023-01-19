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
 * @file apex_config.py 
 * @brief 
 * APEX Configuration object.
 */


'''

import xml.etree.ElementTree as ET
from xml.dom import minidom

from ..apex_API_enums import *

class ApexConfig():
    def __init__(self):
        self.__base_sample_rate = TMSiBaseSampleRate.Decimal
        self.__channels = []
        self.__impedance_channels = []
        self.__impedance_limit = 0
        self.__live_impedance = TMSiLiveImpedance.Off
        self.__sampling_frequency = 0

    def export_to_xml(self, filename):
        try:
            root = ET.Element("ApexConfig")
            xml_device = ET.SubElement(root, "Device")
            ET.SubElement(xml_device, "BaseSampleRate").text = str(self.__base_sample_rate)
            ET.SubElement(xml_device, "ImpedanceLimit").text = str(self.__impedance_limit)
            ET.SubElement(xml_device, "LiveImpedance").text = str(self.__live_impedance)
            xml_channels = ET.SubElement(root, "Channels")
            for idx, channel in enumerate(self.__channels):
                xml_channel = ET.SubElement(xml_channels, "Channel")
                ET.SubElement(xml_channel, "ChanIdx").text = str(idx)
                ET.SubElement(xml_channel, "AltChanName").text = channel.get_channel_name()
                ET.SubElement(xml_channel, "ReferenceStatus").text = str(channel.is_reference()) 
            xml_data = ApexConfig.__prettify(root)
            xml_file = open(filename, "w")
            xml_file.write(xml_data)
            return True
        except:
            return False

    def import_from_xml(self, filename):
        try:
            tree = ET.parse(filename)
            root = tree.getroot()
            for elem in root:
                for subelem in elem:
                    if elem.tag == "Device":
                        if subelem.tag == "BaseSampleRate":
                            self.__base_sample_rate = int(subelem.text)
                        if subelem.tag == "ImpedanceLimit":
                            self.__impedance_limit = int(subelem.text)
                        if subelem.tag == "LiveImpedance":
                            self.__live_impedance = int(subelem.text)
                    if elem.tag == "Channels":
                        if subelem.tag == "Channel":
                            found = False
                            idx = subelem.find("ChanIdx")
                            if idx is None:
                                continue
                            idx = int(idx.text)
                            self.__channels[idx].set_device_channel_names(
                                alternative_channel_name = subelem.find("AltChanName").text
                            )
                            reference = subelem.find("ReferenceStatus").text
                            if reference != 'None':
                                self.__channels[idx].set_device_reference(int(reference))
                            else:
                                self.__channels[idx].set_device_reference(0)
            return True
        except:
            return False
    
    def get_channels(self):
        return self.__channels

    def get_impedance_channels(self):
        return self.__impedance_channels
    
    def get_impedance_limit(self):
        return self.__impedance_limit

    def get_live_impedance(self):
        return self.__live_impedance

    def get_sample_rate(self):
        return self.__base_sample_rate

    def get_sampling_frequency(self):
        return self.__sampling_frequency
    
    def set_channels(self, channels):
        self.__channels = channels

    def set_impedance_channels(self, channels):
        self.__impedance_channels = channels
        
    def set_device_sampling_config(self, device_sampling_config):
        self.__base_sample_rate = device_sampling_config.BaseSampleRate
        self.__impedance_limit = device_sampling_config.ImpedanceLimit
        self.__live_impedance = device_sampling_config.LiveImpedance

    def set_sampling_frequency(self, sampling_frequency):
        self.__sampling_frequency = sampling_frequency

    def __prettify(elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

