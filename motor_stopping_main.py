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
 * @file ${example_stream_lsl.py} 
 * @brief This example shows the functionality to stream to LSL.
 *
 */


"""
import pathlib
import sys

# modules_dir = join(this_dir, '..') # directory with all modules
# sys.path.append(modules_dir)

# from PySide2 import QtWidgets
# from TMSiFileFormats.file_writer import FileFormat, FileWriter
# from TMSiPlotters.gui import PlottingGUI
# from TMSiPlotters.plotters import PlotterFormat
# from TMSiSDK import tmsi_device
# from TMSiSDK.device import DeviceInterfaceType, DeviceState
# from TMSiSDK.error import DeviceErrorLookupTable, TMSiError, TMSiErrorCode

import motor_stopping_timeflux


def main() -> None:
    this_dir = pathlib.Path(__file__).parent  # directory of this file
    data_dir = this_dir / "data"  # directory with all measurements
    assert data_dir.is_dir()

    motor_stopping_timeflux.run(config_file="timeflux_decoding.yaml")

    # try:
    # # Initialise the TMSi-SDK first before starting using it
    # tmsi_device.initialize()
    #     # Execute a device discovery. This returns a list of device-objects for every discovere device.
    #     discovery_list = tmsi_device.discover(
    #         tmsi_device.DeviceType.saga,
    #         DeviceInterfaceType.docked,
    #         DeviceInterfaceType.usb
    #         )

    #     if (len(discovery_list) > 0):
    #         raise ValueError(
    #             "More than one TMSi device found. Please check your"
    #             f" connections. Found: {discovery_list}."
    #         )

    #     # Get the handle to the first discovered device.
    #     device = discovery_list[0]
    #     # Open a connection to the SAGA-system
    #     device.open()
    #     # Initialise the lsl-stream
    #     stream = FileWriter(FileFormat.lsl, "RawSAGA")
    #     # Define the handle to the device
    #     stream.open(device)

    #     # Check if there is already a plotter application in existence
    #     plotter_app = QtWidgets.QApplication.instance()

    #     # Initialise the plotter application if there is no other plotter application
    #     if not plotter_app:
    #         plotter_app = QtWidgets.QApplication(sys.argv)

    #     # Define the GUI object and show it
    #     # The channel selection argument states which channels need to be displayed initially by the GUI
    #     plot_window = PlottingGUI(
    #         plotter_format = PlotterFormat.signal_viewer,
    #         figurename = 'A RealTimePlot',
    #         device = device,
    #         channel_selection = [0,1,2]
    #     )
    #     plot_window.show()

    #     # Enter the event loop
    #     plotter_app.exec_()

    #     # Quit and delete the Plotter application
    #     QtWidgets.QApplication.quit()
    #     del plotter_app

    #     # Close the file writer after GUI termination
    #     stream.close()

    #     # Close the connection to the SAGA device
    #     device.close()

    # except TMSiError as e:
    #     print("!!! TMSiError !!! : ", e.code)
    #     if e.code == TMSiErrorCode.device_error:
    #         print("  => device error : ", hex(device.status.error))
    #         DeviceErrorLookupTable(hex(device.status.error))

    # finally:
    #     # Close the connection to the device when the device is opened
    #     if device.status.state == DeviceState.connected:
    #         device.close()


if __name__ == "__main__":
    main()
