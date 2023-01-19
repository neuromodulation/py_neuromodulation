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
 * @file ${xml_saga_config.py} 
 * @brief Reading/Writing SAGA configuration via XML
 *
 */


'''

from xml.dom import minidom
import xml.etree.ElementTree as ET

from .saga_types import SagaConfig, SagaChannel

def __prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def xml_write_config(filename, saga_config):
    """Writes a device configuration to the specified file.

       Args:
           'filename' Location of the xml-file to write

           'saga_config' device configuration to write

       Returns:
           True if successfull, False otherwise.
    """
    try:
        root = ET.Element("DeviceConfig")

        # 1. Write the device configuration properties
        xml_device = ET.SubElement(root, "Device")

        ET.SubElement(xml_device, "BaseSampleRateHz").text = str(saga_config._base_sample_rate)
        ET.SubElement(xml_device, "ConfiguredInterface").text = str(saga_config._configured_interface)
        ET.SubElement(xml_device, "Triggers").text = str(saga_config._triggers)
        ET.SubElement(xml_device, "ReferenceMethod").text = str(saga_config._reference_method)
        ET.SubElement(xml_device, "AutoReferenceMethod").text = str(saga_config._auto_reference_method)
        ET.SubElement(xml_device, "DRSyncOutDiv").text = str(saga_config._dr_sync_out_divider)
        ET.SubElement(xml_device, "DRSyncOutDutyCycl").text = str(saga_config._dr_sync_out_duty_cycle)
        ET.SubElement(xml_device, "RepairLogging").text = str(saga_config._repair_logging)

        # 2. Write the channel list
        xml_channels = ET.SubElement(root, "Channels")
        for idx, saga_channel in enumerate(saga_config._channels):

            xml_channel = ET.SubElement(xml_channels, "Channel")

            ET.SubElement(xml_channel, "ChanNr").text = str(idx)
            ET.SubElement(xml_channel, "ChanDivider").text = str(saga_channel.chan_divider)
            ET.SubElement(xml_channel, "AltChanName").text = saga_channel.alt_name

        xml_data = __prettify(root)

        xml_file = open(filename, "w")
        xml_file.write(xml_data)

        return True

    except:
        return False

def xml_read_config(filename) :
    """Read a device configuration from the specified file.

       Args:
           'filename' Location of the xml-file to read

       Returns:
           if successfull :
               True and a 'SagaConfig'-object with the read configuration.
           otherwise :
               False and a default 'SagaConfig'-object
    """
    result = True
    saga_config = SagaConfig()

    try:
        tree = ET.parse(filename)
        root = tree.getroot()

        for elem in root:
            # print(len(elem))
            for subelem in elem:
                if elem.tag == "Device":
                    if subelem.tag == "BaseSampleRateHz":
                        saga_config._base_sample_rate = int(subelem.text)
                    if subelem.tag == "ConfiguredInterface":
                        saga_config._configured_interface = int(subelem.text)
                    if subelem.tag == "Triggers":
                        saga_config._triggers = int(subelem.text)
                    if subelem.tag == "ReferenceMethod":
                        saga_config._reference_method = int(subelem.text)
                    if subelem.tag == "AutoReferenceMethod":
                        saga_config._auto_reference_method = int(subelem.text)
                    if subelem.tag == "DRSyncOutDiv":
                        saga_config._dr_sync_out_divider = int(subelem.text)
                    if subelem.tag == "DRSyncOutDutyCycl":
                        saga_config._dr_sync_out_duty_cycle = int(subelem.text)
                    if subelem.tag == "RepairLogging":
                        saga_config._repair_logging = int(subelem.text)
                if elem.tag == "Channels":
                    if subelem.tag == "Channel":
                        saga_config._num_channels += 1
                        channel = SagaChannel()
                        for subsubelem in subelem:
                            if subsubelem.tag == "ChanDivider":
                                channel.chan_divider = int(subsubelem.text)
                            if subsubelem.tag == "AltChanName":
                                channel.alt_name = subsubelem.text

                        saga_config._channels.append(channel)
    except:
        result = False
        saga_config = None

    return result, saga_config




