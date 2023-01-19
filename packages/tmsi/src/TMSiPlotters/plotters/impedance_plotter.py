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
 * @file ${impedance_plotter.py} 
 * @brief Plotter object that displays impedance values in real-time.
 *
 */


"""

from os.path import join, dirname, realpath, normpath, exists
from PySide2 import QtGui, QtCore, QtWidgets
import numpy as np
import pyqtgraph as pg
import time
import queue
import pandas as pd
import math
import datetime
import sys


Plotter_dir = dirname(realpath(__file__))  # directory of this file
measurements_dir = join(
    Plotter_dir, "../../measurements"
)  # directory with all measurements
modules_dir = normpath(join(Plotter_dir, "../.."))  # directory with all modules

from TMSiSDK import tmsi_device
from TMSiSDK import sample_data_server
from TMSiSDK import sample_data
from TMSiSDK.device import DeviceInterfaceType, MeasurementType

from apex_sdk.device.tmsi_device import TMSiDevice
from apex_sdk.sample_data_server.sample_data_server import (
    SampleDataServer as ApexSampleDataServer,
)


class ImpedanceViewer:
    """Class that creates a GUI to display the impedance values in a gridded
    layout.
    """

    def __init__(
        self,
        gui_handle,
        device,
        layout="normal",
        file_storage=False,
        grid_type="none",
    ):
        """Setting up the GUI's elements"""
        # Pass the device handle to the GUI
        self.device = device

        # Path to store the impedances, if a path is specified
        self._save_impedances = file_storage

        self.gui_handle = gui_handle
        self.RealTimePlotWidget = self.gui_handle.RealTimePlotWidget

        self.grid_type = grid_type
        self._disable_channels = []

        # Number of impedance channels
        if isinstance(self.device, TMSiDevice):
            self.n_chan = len(self.device.get_device_impedance_channels())
        else:
            self.n_chan = len(self.device.imp_channels)

        # Set up UI and thread
        self.initUI(layout)
        self.setupThread()

    def initUI(self, layout):
        """Method responsible for constructing the basic elements in the plot.
        All viewboxes have a set size so that the information can be displayed
        correctly.
        """
        # Set view settings
        self.RealTimePlotWidget.setBackground("w")
        self.RealTimePlotWidget.showMaximized()

        # Add viewbox for the legend
        self.RealTimePlotWidget.vb_legend = self.RealTimePlotWidget.addViewBox()
        legend = pg.LegendItem()
        legend.setParentItem(self.RealTimePlotWidget.vb_legend)

        # Add plot window for the channels
        self.RealTimePlotWidget.window = self.RealTimePlotWidget.addPlot()

        if layout != "head":
            self.RealTimePlotWidget.window.getViewBox().invertY(True)

            if self.n_chan <= 34:
                self.RealTimePlotWidget.window.setAspectLocked(
                    lock=True, ratio=0.6
                )
            else:
                self.RealTimePlotWidget.window.setAspectLocked(
                    lock=True, ratio=1
                )
        else:
            self.RealTimePlotWidget.window.setAspectLocked(lock=True, ratio=1)

        # Add viewbox for the list of values
        self.RealTimePlotWidget.vb_list = self.RealTimePlotWidget.addViewBox()
        self.RealTimePlotWidget.vb_list.setMaximumSize(500, 150000)

        # Generate the legend by using dummy plots
        self.legend_entries = []
        legend_spots = self._generate_legend()
        for i in range(len(legend_spots)):

            lg_plt = pg.ScatterPlotItem(
                pos=[(0, 0), (0, 0)],
                size=20,
                pen=legend_spots[i]["pen"],
                brush=legend_spots[i]["brush"],
                name=legend_spots[i]["name"],
            )
            legend.addItem(lg_plt, legend_spots[i]["name"])
            lg_plt.clear()

        # Ensure legend is displayed in the top left of the viewbox
        self.RealTimePlotWidget.vb_legend.setMaximumWidth(
            legend.boundingRect().width() + 10
        )
        legend.anchor((0, 0), (0, 0))

        # Write the ticks to the plot
        self.RealTimePlotWidget.window.hideAxis("left")
        self.RealTimePlotWidget.window.hideAxis("bottom")

        # Disable auto-scaling and menu
        self.RealTimePlotWidget.window.hideButtons()
        self.RealTimePlotWidget.window.setMenuEnabled(False)
        self.RealTimePlotWidget.window.setMouseEnabled(x=False, y=False)
        self.RealTimePlotWidget.vb_legend.setMouseEnabled(x=False, y=False)
        self.RealTimePlotWidget.vb_list.setMouseEnabled(x=False, y=False)

        # Initialise the standard format for the different indicators
        self.spots = [
            {
                "pos": (0, 0),
                "size": 20,
                "pen": "k",
                "brush": QtGui.QBrush(QtGui.QColor(128, 128, 128)),
            }
            for i in range(self.n_chan)
        ]

        # Get channel names and units
        if isinstance(self.device, TMSiDevice):
            chs = [
                [i.get_channel_name(), i.get_channel_unit_name()]
                for i in self.device.get_device_impedance_channels()
            ]

        if layout == "head":
            # read channel locations
            chLocs = pd.read_csv(
                join(
                    modules_dir, "TMSiSDK", "_resources", "EEGchannelsTMSi.txt"
                ),
                sep="\t",
                header=None,
            )
            chLocs.columns = ["name", "radius", "theta"]

            # Plot a circle
            theta = np.arange(0, 2.02 * math.pi, math.pi / 50)
            x_circle = 0.5 * np.cos(theta)
            y_circle = 0.5 * np.sin(theta)
            self.h = pg.PlotCurveItem()
            self.h.setData(
                x_circle, y_circle, pen=pg.mkPen((165, 165, 165), width=5)
            )
            self.RealTimePlotWidget.window.addItem(self.h)

            # Plot a nose
            y_nose = np.array([x_circle[2], 0.55, x_circle[-3]])
            x_nose = np.array([y_circle[2], 0, y_circle[-3]])
            self.n = pg.PlotCurveItem()
            self.n.setData(
                x_nose, y_nose, pen=pg.mkPen((165, 165, 165), width=5)
            )
            self.RealTimePlotWidget.window.addItem(self.n)

            # Plot ears
            x_ears = np.array(
                [0.49, 0.51, 0.52, 0.53, 0.54, 0.54, 0.55, 0.53, 0.51, 0.485]
            )
            y_ears = np.array(
                [
                    0.10,
                    0.1175,
                    0.1185,
                    0.1145,
                    0.0955,
                    -0.0055,
                    -0.0930,
                    -0.1315,
                    -0.1385,
                    -0.12,
                ]
            )
            self.e = pg.PlotCurveItem()
            self.e.setData(
                x_ears, y_ears, pen=pg.mkPen((165, 165, 165), width=5)
            )
            self.RealTimePlotWidget.window.addItem(self.e)
            self.e = pg.PlotCurveItem()
            self.e.setData(
                -x_ears, y_ears, pen=pg.mkPen((165, 165, 165), width=5)
            )
            self.RealTimePlotWidget.window.addItem(self.e)

            # Set the position for each indicator
            for i in range(self.n_chan):
                if i == 0:
                    if isinstance(self.device, TMSiDevice):
                        self.spots[i]["pos"] = (0, -0.6)
                    else:
                        self.spots[i]["pos"] = (-0.05, -0.6)
                elif i == self.n_chan - 1 and not isinstance(
                    self.device, TMSiDevice
                ):
                    self.spots[i]["pos"] = (0.05, -0.6)
                else:
                    if isinstance(self.device, TMSiDevice):
                        idx_ch = np.where(
                            "'" + chs[i][0] + "'" == chLocs["name"]
                        )[0][0]

                        x = chLocs["radius"].values[idx_ch] * np.sin(
                            np.deg2rad(chLocs["theta"].values[idx_ch])
                        )
                        y = chLocs["radius"].values[idx_ch] * np.cos(
                            np.deg2rad(chLocs["theta"].values[idx_ch])
                        )
                    else:
                        x = chLocs["radius"].values[i - 1] * np.sin(
                            np.deg2rad(chLocs["theta"].values[i - 1])
                        )
                        y = chLocs["radius"].values[i - 1] * np.cos(
                            np.deg2rad(chLocs["theta"].values[i - 1])
                        )

                    self.spots[i]["pos"] = (x, y)

                # Place the name of each channel below the respective indicator
                if isinstance(self.device, TMSiDevice):
                    text = f"{chs[i][0]: ^10}"
                else:
                    text = f"{self.device.imp_channels[i].name: ^10}"
                t_item = pg.TextItem(text, (0, 0, 0), anchor=(0, 0))
                t_item.setPos(
                    self.spots[i]["pos"][0] - 0.03,
                    self.spots[i]["pos"][1] - 0.02,
                )
                self.RealTimePlotWidget.window.addItem(t_item)

            # Generate the channel conversion list
            self.channel_conversion_list = np.arange(0, self.n_chan, dtype=int)

        else:
            self.channel_conversion_list = (
                self.gui_handle.channel_conversion_list
            )

            # Insert dummy channels
            self.dummy_chan = []
            if "11" in self.grid_type:
                if not self.grid_type[-1] == "2":
                    self.dummy_chan.append(10)
                if not self.grid_type[-1] == "1":
                    dummy_chan = max(self.channel_conversion_list)
                    self.dummy_chan.append(dummy_chan)

            # Insert PGND channel in channel conversion list
            self.channel_conversion_list = np.hstack(
                (
                    self.channel_conversion_list,
                    max(self.gui_handle.channel_conversion_list) + 1,
                )
            )
            # In case of 64 channel device with 32 channel grid, insert channels 33-64
            if (
                len(self.device.imp_channels) > 34
                and max(self.channel_conversion_list) < 34
            ):
                self.channel_conversion_list = np.hstack(
                    (self.channel_conversion_list, np.arange(34, 66))
                )

            row_count = -1
            c_offset = 0

            if "11" in self.grid_type:
                # Set the position for each indicator
                for i in range(len(self.device.imp_channels)):
                    if i == 0:
                        if len(self.device.imp_channels) > 34:
                            self.spots[i]["pos"] = (3, 8)
                        else:
                            self.spots[i]["pos"] = (3, 4)
                    elif i == len(self.device.imp_channels) - 1:
                        if len(self.device.imp_channels) > 34:
                            self.spots[i]["pos"] = (4, 8)
                        else:
                            self.spots[i]["pos"] = (4, 4)
                    elif i - 1 in self.dummy_chan:
                        row_count += 1
                        self.spots[i]["pos"] = (0, row_count)
                        c_offset = i - 1
                    elif (i - 1 - c_offset) % 11 == 0:
                        row_count += 1
                        self.spots[i]["pos"] = (0, row_count)
                    else:
                        self.spots[i]["pos"] = (
                            ((i - c_offset - 1) % 11),
                            row_count,
                        )
            else:
                # Set the position for each indicator
                for i in range(self.n_chan):
                    if i == 0:
                        if self.n_chan > 34:
                            self.spots[i]["pos"] = (3, 8)
                        else:
                            self.spots[i]["pos"] = (3, 4)
                    elif i == self.n_chan - 1 and not isinstance(
                        self.device, TMSiDevice
                    ):
                        if self.n_chan > 34:
                            self.spots[i]["pos"] = (4, 8)
                        else:
                            self.spots[i]["pos"] = (4, 4)
                    elif (i - 1) % 8 == 0:
                        row_count += 1
                        self.spots[i]["pos"] = (((i - 1) % 8), row_count)
                    else:
                        if i == 0:
                            if len(self.device.imp_channels) > 34:
                                self.spots[i]["pos"] = (3, 8)
                            else:
                                self.spots[i]["pos"] = (3, 4)
                        elif i == len(self.device.imp_channels) - 1:
                            if len(self.device.imp_channels) > 34:
                                self.spots[i]["pos"] = (4, 8)
                            else:
                                self.spots[i]["pos"] = (4, 4)
                        elif (i - 1) % 8 == 0:
                            row_count += 1
                            self.spots[i]["pos"] = (((i - 1) % 8), row_count)
                        else:
                            self.spots[i]["pos"] = (((i - 1) % 8), row_count)

            for i in range(self.n_chan):
                # Place the name of each channel below the respective indicator
                if isinstance(self.device, TMSiDevice):
                    text = f"{chs[self.channel_conversion_list[i]][0]: ^10}"
                else:
                    text = f"{self.device.imp_channels[self.channel_conversion_list[i]].name: ^10}"
                t_item = pg.TextItem(text, (128, 128, 128), anchor=(0, 0))
                t_item.setPos(
                    self.spots[i]["pos"][0] - 0.25,
                    self.spots[i]["pos"][1] + 0.1,
                )
                self.RealTimePlotWidget.window.addItem(t_item)

        # Add all indicators to the plot
        self.c = pg.ScatterPlotItem(self.spots)
        self.RealTimePlotWidget.window.addItem(self.c)

        # Create a list with impedance values to display next to the plot
        self.text_items = []
        for i in range(self.n_chan):
            # Display 33 names per list (66 impedance channels in SAGA64+)
            list_split_idx = 33
            num_column = np.floor(i / list_split_idx)
            value = 5000
            if isinstance(self.device, TMSiDevice):
                text = f"{chs[self.channel_conversion_list[i]][0]}\t{value:>4}\t{chs[self.channel_conversion_list[i]][1][0]}"
            else:
                text = f"{self.device.imp_channels[self.channel_conversion_list[i]].name}\t{value:>4}\t{self.device.imp_channels[self.channel_conversion_list[i]].unit_name}"
            t_item = pg.TextItem(
                text,
                (0, 0, 0),
                anchor=(
                    -num_column * 1.2,
                    -i * 0.9
                    + list_split_idx * 0.9 * np.floor(i / list_split_idx),
                ),
            )
            self.text_items.append(t_item)
            self.RealTimePlotWidget.vb_list.addItem(t_item)

    @QtCore.Slot(object)
    def update_plot(self, data):
        """Method that updates the indicators according to the measured impedance values"""
        for i in range(len(self.spots)):
            if isinstance(self.device, TMSiDevice):
                if (
                    self.device.get_device_impedance_channels()[
                        self.channel_conversion_list[i]
                    ].get_channel_name()
                    in self._disable_channels
                ):
                    self.spots[i]["brush"] = QtGui.QBrush(
                        QtGui.QColor(175, 175, 175)
                    )
                else:
                    self.spots[i]["brush"] = QtGui.QBrush(
                        self._lookup_table(data[i])
                    )
                text = f"{self.device.get_device_impedance_channels()[self.channel_conversion_list[i]].get_channel_name()}\t{data[i]:>4}\t{self.device.get_device_impedance_channels()[self.channel_conversion_list[i]].get_channel_unit_name()[0]}"
            else:
                if (
                    self.device.imp_channels[
                        self.channel_conversion_list[i]
                    ].name
                    in self._disable_channels
                ):
                    self.spots[i]["brush"] = QtGui.QBrush(
                        QtGui.QColor(175, 175, 175)
                    )
                else:
                    self.spots[i]["brush"] = QtGui.QBrush(
                        self._lookup_table(data[i])
                    )
                text = f"{self.device.imp_channels[self.channel_conversion_list[i]].name}\t{data[i]:>4}\t{self.device.imp_channels[self.channel_conversion_list[i]].unit_name}"
            self.text_items[i].setText(text)
        self.c.setData(self.spots)

    def _lookup_table(self, value):
        """Look up table to convert impedances to color coding is device specific"""
        if isinstance(self.device, TMSiDevice):
            return self._lookup_table_APEX(value)
        else:
            return self._lookup_table_SAGA(value)

    def _lookup_table_SAGA(self, value):
        """Look up table to convert impedances to color coding"""
        if value < 5:
            color_code = QtGui.QColor(0, 255, 0)
        elif value >= 5 and value < 10:
            color_code = QtGui.QColor(0, 204, 0)
        elif value >= 10 and value < 30:
            color_code = QtGui.QColor(0, 153, 0)
        elif value >= 30 and value < 50:
            color_code = QtGui.QColor(0, 102, 0)
        elif value >= 50 and value < 100:
            color_code = QtGui.QColor(255, 255, 0)
        elif value >= 100 and value < 200:
            color_code = QtGui.QColor(204, 128, 0)
        elif value >= 200 and value < 400:
            color_code = QtGui.QColor(255, 0, 0)
        elif value >= 400 and value < 500:
            color_code = QtGui.QColor(153, 0, 0)
        elif value == 500:
            color_code = QtGui.QColor(0, 0, 0)
        elif value == 5000:
            color_code = QtGui.QColor(175, 175, 175)
        elif value == 5100:
            color_code = QtGui.QColor(204, 0, 102)
        elif value == 5200:
            color_code = QtGui.QColor(0, 0, 179)
        return color_code

    def _lookup_table_APEX(self, value):
        """Look up table to convert impedances to color coding"""
        if value < 5:
            color_code = QtGui.QColor(0, 255, 0)
        elif value >= 5 and value < 10:
            color_code = QtGui.QColor(0, 204, 0)
        elif value >= 10 and value < 30:
            color_code = QtGui.QColor(0, 153, 0)
        elif value >= 30 and value < 50:
            color_code = QtGui.QColor(0, 102, 0)
        elif value >= 50 and value < 100:
            color_code = QtGui.QColor(255, 255, 0)
        elif value >= 100 and value < 200:
            color_code = QtGui.QColor(204, 128, 0)
        elif value >= 200 and value < 400:
            color_code = QtGui.QColor(255, 0, 0)
        elif value >= 400 and value < 1000:
            color_code = QtGui.QColor(153, 0, 0)
        else:
            color_code = QtGui.QColor(38, 38, 38)
        return color_code

    def _generate_legend(self):
        """Generate the device specific legend"""
        if isinstance(self.device, TMSiDevice):
            return self._generate_APEX_legend()
        else:
            return self._generate_SAGA_legend()

    def _generate_SAGA_legend(self):
        """Method that generates the dummy samples needed to plot the legend"""
        legend_spots = [
            {
                "pos": (0, 0),
                "size": 10,
                "pen": "k",
                "brush": QtGui.QBrush(),
                "name": "",
            }
            for i in range(12)
        ]
        legend_spots[0]["name"] = "0 - 5 k\u03A9"
        legend_spots[0]["brush"] = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        legend_spots[1]["name"] = "5 - 10 k\u03A9"
        legend_spots[1]["brush"] = QtGui.QBrush(QtGui.QColor(0, 204, 0))
        legend_spots[2]["name"] = "10 - 30 k\u03A9"
        legend_spots[2]["brush"] = QtGui.QBrush(QtGui.QColor(0, 153, 0))
        legend_spots[3]["name"] = "30 - 50 k\u03A9"
        legend_spots[3]["brush"] = QtGui.QBrush(QtGui.QColor(0, 102, 0))
        legend_spots[4]["name"] = "50 - 100 k\u03A9"
        legend_spots[4]["brush"] = QtGui.QBrush(QtGui.QColor(255, 255, 0))
        legend_spots[5]["name"] = "100 - 200 k\u03A9"
        legend_spots[5]["brush"] = QtGui.QBrush(QtGui.QColor(204, 128, 0))
        legend_spots[6]["name"] = "200 - 400 k\u03A9"
        legend_spots[6]["brush"] = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        legend_spots[7]["name"] = "400 - 500 k\u03A9"
        legend_spots[7]["brush"] = QtGui.QBrush(QtGui.QColor(153, 0, 0))
        legend_spots[8]["name"] = "≥ 500 k\u03A9 / Not connected"
        legend_spots[8]["brush"] = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        legend_spots[9]["name"] = "Disabled"
        legend_spots[9]["brush"] = QtGui.QBrush(QtGui.QColor(175, 175, 175))
        legend_spots[10]["name"] = "Odd/Even error"
        legend_spots[10]["brush"] = QtGui.QBrush(QtGui.QColor(204, 0, 102))
        legend_spots[11]["name"] = "PGND disconnected"
        legend_spots[11]["brush"] = QtGui.QBrush(QtGui.QColor(0, 0, 179))
        return legend_spots

    def _generate_APEX_legend(self):
        """Method that generates the dummy samples needed to plot the legend"""
        legend_spots = [
            {
                "pos": (0, 0),
                "size": 10,
                "pen": "k",
                "brush": QtGui.QBrush(),
                "name": "",
            }
            for i in range(9)
        ]
        legend_spots[0]["name"] = "0 - 5 k\u03A9"
        legend_spots[0]["brush"] = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        legend_spots[1]["name"] = "5 - 10 k\u03A9"
        legend_spots[1]["brush"] = QtGui.QBrush(QtGui.QColor(0, 204, 0))
        legend_spots[2]["name"] = "10 - 30 k\u03A9"
        legend_spots[2]["brush"] = QtGui.QBrush(QtGui.QColor(0, 153, 0))
        legend_spots[3]["name"] = "30 - 50 k\u03A9"
        legend_spots[3]["brush"] = QtGui.QBrush(QtGui.QColor(0, 102, 0))
        legend_spots[4]["name"] = "50 - 100 k\u03A9"
        legend_spots[4]["brush"] = QtGui.QBrush(QtGui.QColor(255, 255, 0))
        legend_spots[5]["name"] = "100 - 200 k\u03A9"
        legend_spots[5]["brush"] = QtGui.QBrush(QtGui.QColor(204, 128, 0))
        legend_spots[6]["name"] = "200 - 400 k\u03A9"
        legend_spots[6]["brush"] = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        legend_spots[7]["name"] = "400 - 1000 k\u03A9"
        legend_spots[7]["brush"] = QtGui.QBrush(QtGui.QColor(153, 0, 0))
        legend_spots[8]["name"] = "≥ 1000 k\u03A9 / Not connected"
        legend_spots[8]["brush"] = QtGui.QBrush(QtGui.QColor(38, 38, 38))

        return legend_spots

    def _update_active_channels(self):
        self._disable_channels = []
        # Update the channel selection based on the clicked checkboxes
        for i in range(np.size(self.gui_handle._checkboxes, 0)):
            if not self.gui_handle._checkboxes[i].isChecked():
                self._disable_channels.append(
                    self.device.channels[
                        self.gui_handle.active_channel_conversion_list[i]
                    ].name
                )

    def setupThread(self):
        """Method that initialises the sampling thread of the device"""
        # Create a Thread
        self.thread = QtCore.QThread()
        # Instantiate the worker class
        self.worker = SamplingThread(self)

        # Move the worker to a Thread
        self.worker.moveToThread(self.thread)

        # Connect signals to slots
        self.thread.started.connect(self.worker.update_samples)
        self.worker.output.connect(self.update_plot)


class SamplingThread(QtCore.QObject):
    """Class responsible for sampling the data from the device"""

    # Initialise the ouptut object
    output = QtCore.Signal(object)

    def __init__(self, main_class):
        QtCore.QObject.__init__(self)
        # Access initialised values from the GUI class
        self.device = main_class.device
        self._save_impedances = main_class._save_impedances

        self.grid_type = main_class.grid_type
        self.channel_conversion_list = main_class.channel_conversion_list

        # Prepare Queue
        self.q_sample_sets = queue.Queue(1000)

        # Register the consumer to the sample server
        if isinstance(self.device, TMSiDevice):
            ApexSampleDataServer().register_consumer(
                self.device.get_id(), self.q_sample_sets
            )
        else:
            sample_data_server.registerConsumer(
                self.device.id, self.q_sample_sets
            )

        # # Start measurement
        self.sampling = True

    @QtCore.Slot()
    def update_samples(self):
        """Method that retrieves the sample data from the device. The method
        gives the impedance value as output
        """
        while self.sampling:
            while not self.q_sample_sets.empty():
                sd = self.q_sample_sets.get()
                self.q_sample_sets.task_done()

                # Retrieve the data from the queue and write it to a SampleSet object
                for i in range(sd.num_sample_sets):
                    sample_set = sample_data.SampleSet(
                        sd.num_samples_per_sample_set,
                        sd.samples[
                            i
                            * sd.num_samples_per_sample_set : (i + 1)
                            * sd.num_samples_per_sample_set
                        ],
                    )

                # Use the final measured impedance value and convert to integer value
                impedance_values = [int(x) for x in sample_set.samples]

                if isinstance(self.device, TMSiDevice):
                    impedance_values = [
                        impedance_values[i * 2]
                        for i in self.channel_conversion_list
                    ]
                else:
                    impedance_values = [
                        impedance_values[i]
                        for i in self.channel_conversion_list
                    ]

                self.impedance_values = impedance_values

                # Output sample data
                self.output.emit(impedance_values)

            # Pause the thread so that the update does not happen too fast
            time.sleep(1)

    def stop(self):
        """Method that is executed when the thread is terminated.
        This stop event stops the measurement and closes the connection to
        the device.
        """
        self.sampling = False

        if self._save_impedances:
            store_imp = []

            for i in range(len(self.impedance_values)):
                if isinstance(self.device, TMSiDevice):
                    store_imp.append(
                        f"{self.device.get_device_impedance_channels()[i].get_channel_name()}\t{self.impedance_values[self.channel_conversion_list[i]]}\t{self.device.get_device_impedance_channels()[self.channel_conversion_list[i]].get_channel_unit_name()[0]}"
                    )
                else:
                    store_imp.append(
                        f"{self.device.imp_channels[i].name}\t{self.impedance_values[self.channel_conversion_list[i]]}\t{self.device.imp_channels[self.channel_conversion_list[i]].unit_name}"
                    )

            now = datetime.datetime.now()
            filetime = now.strftime("%Y%m%d_%H%M%S")
            filename = self._save_impedances + "-" + filetime

            with open(filename + ".txt", "w") as f:
                for item in store_imp:
                    f.write(item + "\n")


if __name__ == "__main__":
    # Initialise the TMSi-SDK first before starting using it
    tmsi_device.initialize()

    # Create the device object to interface with the SAGA-system.
    dev = tmsi_device.create(
        tmsi_device.DeviceType.saga,
        DeviceInterfaceType.docked,
        DeviceInterfaceType.usb,
    )

    # Find and open a connection to the SAGA-system and print its serial number
    dev.open()
    print("handle 1 " + str(dev.info.ds_serial_number))

    # Initialise the application
    app = QtGui.QApplication(sys.argv)
    # Define the GUI object and show it
    window = ImpedancePlot(
        figurename="An Impedance Plot", device=dev, layout="normal"
    )
    window.show()

    # Enter the event loop
    # sys.exit(app.exec_())
    app.exec_()
    dev.close()
