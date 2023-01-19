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
 * @file ${signal_plotter.py} 
 * @brief Plotter object that displays incoming sample data in real-time.
 *
 */


"""

from PySide2 import QtGui, QtCore, QtWidgets
from PySide2.QtCore import Qt
import numpy as np
import pyqtgraph as pg
import time
import queue
from copy import copy
import sys

from os.path import join, dirname, realpath, normpath, exists

Plotter_dir = dirname(realpath(__file__))  # directory of this file
measurements_dir = join(
    Plotter_dir, "../../measurements"
)  # directory with all measurements
modules_dir = normpath(join(Plotter_dir, "../.."))  # directory with all modules

from TMSiSDK import tmsi_device
from TMSiSDK import sample_data_server
from TMSiSDK.device import DeviceInterfaceType, ChannelType

from apex_sdk.device.tmsi_device import TMSiDevice
from apex_sdk.sample_data_server.sample_data_server import (
    SampleDataServer as ApexSampleDataServer,
)


class SignalViewer:
    """A GUI that displays the signals on the screen. The GUI handles the
    incoming data and is able to apply scaling. Furthermore, the GUI handles
    closing the device when it is closed.
    """

    def __init__(self, window, device, filter_app=None, grid_type="none"):
        """Setting up the GUI's elements."""
        self.gui_handle = window
        self.RealTimePlotWidget = self.gui_handle.RealTimePlotWidget

        # Pass the device handle so that it is accesible to the GUI
        self.device = device
        self.filter_app = filter_app
        self.grid_type = grid_type

        # Set up UI and thread
        self.initUI()
        self.live_impedance = False
        if isinstance(self.device, TMSiDevice):
            if self.device.get_device_sampling_config().LiveImpedance:
                self.init_impedance_table()
                self.live_impedance = True
        self.setupThread()

    def initUI(self):
        """Method responsible for constructing the basic elements in the plot"""
        # Set view settings
        self.RealTimePlotWidget.setBackground("w")
        self.RealTimePlotWidget.window = self.RealTimePlotWidget.addPlot()
        self.RealTimePlotWidget.window.showGrid(x=True, y=True, alpha=0.5)
        self.RealTimePlotWidget.window.setLabel("bottom", "Time", units="sec")
        self.RealTimePlotWidget.window.getViewBox().invertY(True)

        self.channel_conversion_list = (
            self.gui_handle.active_channel_conversion_list
        )

        # Configuration settings
        self.num_channels = np.size(self.gui_handle._channel_selection, 0)
        if isinstance(self.device, TMSiDevice):
            self.active_channels = np.size(self.device.get_device_channels(), 0)
            self.sample_rate = self.device.get_device_sampling_frequency()
        else:
            self.active_channels = np.size(self.device.channels, 0)
            self.sample_rate = self.device.config.get_sample_rate(
                ChannelType.counter
            )
        self.window_size = 5  # seconds

        self._downsampling_factor = int(self.sample_rate / 500)

        # Virtual offset for all channels
        self._plot_offset = 3
        # Dictionary used for storing the scaling factors
        self._plot_diff = [
            {"mean": 0, "diff": 2**31} for i in range(self.num_channels)
        ]
        self._plot_diff[-1]["diff"] = 1
        if isinstance(self.device, TMSiDevice):
            self._plot_diff[-1]["mean"] = 32
        else:
            self._plot_diff[-1]["mean"] = 1024

        if isinstance(self.device, TMSiDevice):
            self.chs = [
                [
                    i.get_channel_name(),
                    i.get_channel_unit_name(),
                    i.get_channel_type(),
                ]
                for i in self.device.get_device_active_channels()
            ]

        tick_list_left = [[]]
        # Create the first instance of the y-axis ticks
        if isinstance(self.device, TMSiDevice):
            for i in range(self.num_channels):
                for j in [-1, 0, 1]:
                    if i == self.num_channels - 1:
                        if not bool(j):
                            tick_list_left[0].append(
                                (
                                    int(self._plot_offset * i),
                                    f"{self.chs[self.gui_handle._channel_selection[i]][0]: <25}",
                                )
                            )
                        else:
                            tick_list_left[0].append(
                                (
                                    int(
                                        self._plot_offset * i
                                        + -j * self._plot_offset / 3
                                    ),
                                    f'{ self._plot_diff[i]["mean"] + (self._plot_diff[i]["diff"] * j ) : .2g}',
                                )
                            )
                    else:
                        if not bool(j):
                            tick_list_left[0].append(
                                (
                                    int(self._plot_offset * i),
                                    f"{self.chs[self.channel_conversion_list[self.gui_handle._channel_selection[i]]][0]: <25}",
                                )
                            )
                        else:
                            tick_list_left[0].append(
                                (
                                    int(
                                        self._plot_offset * i
                                        + -j * self._plot_offset / 3
                                    ),
                                    f'{ self._plot_diff[i]["mean"] + (self._plot_diff[i]["diff"] * j ) : .2g}',
                                )
                            )

            # Display the unit name on the right-side y-axis
            tick_list_right = [
                [
                    (
                        self._plot_offset * i,
                        self.chs[self.gui_handle._channel_selection[i]][1],
                    )
                    for i in range(self.num_channels)
                ]
            ]
        else:
            for i in range(self.num_channels):
                for j in [-1, 0, 1]:
                    if i == self.num_channels - 1:
                        if not bool(j):
                            tick_list_left[0].append(
                                (
                                    int(self._plot_offset * i),
                                    f"{self.device.channels[self.gui_handle._channel_selection[i]].name: <25}",
                                )
                            )
                        else:
                            tick_list_left[0].append(
                                (
                                    int(
                                        self._plot_offset * i
                                        + -j * self._plot_offset / 3
                                    ),
                                    f'{ self._plot_diff[i]["mean"] + (self._plot_diff[i]["diff"] * j ) : .2g}',
                                )
                            )
                    else:
                        if not bool(j):
                            tick_list_left[0].append(
                                (
                                    int(self._plot_offset * i),
                                    f"{self.device.channels[self.channel_conversion_list[self.gui_handle._channel_selection[i]]].name: <25}",
                                )
                            )
                        else:
                            tick_list_left[0].append(
                                (
                                    int(
                                        self._plot_offset * i
                                        + -j * self._plot_offset / 3
                                    ),
                                    f'{ self._plot_diff[i]["mean"] + (self._plot_diff[i]["diff"] * j ) : .2g}',
                                )
                            )

            # Display the unit name on the right-side y-axis
            tick_list_right = [
                [
                    (
                        self._plot_offset * i,
                        self.device.channels[
                            self.gui_handle._channel_selection[i]
                        ].unit_name,
                    )
                    for i in range(self.num_channels)
                ]
            ]

        # Write the ticks to the plot
        self.RealTimePlotWidget.window.getAxis("left").setTicks(tick_list_left)
        self.RealTimePlotWidget.window.getAxis("right").setTicks(
            tick_list_right
        )

        # Disable auto-scaling and menu
        self.RealTimePlotWidget.window.hideButtons()
        self.RealTimePlotWidget.window.setMenuEnabled(False)
        self.RealTimePlotWidget.window.setMouseEnabled(x=False, y=False)

        # Update the ranges to be shown
        self.RealTimePlotWidget.window.showAxis("right")
        self.RealTimePlotWidget.window.setYRange(
            -self._plot_offset / 3,
            (self.num_channels - 2 / 3) * self._plot_offset,
        )
        self.RealTimePlotWidget.window.setXRange(
            0.02 * self.window_size, 0.97 * self.window_size
        )

        # Create curves and set the position of the curve on the y-axis for each channel
        self.curve = []
        for i in range(self.num_channels):
            self.c = pg.PlotCurveItem()
            self.c.setPen(
                color="black", cosmetic=True, joinStyle=QtCore.Qt.MiterJoin
            )
            self.RealTimePlotWidget.window.addItem(self.c)
            self.c.setPos(0, (i) * self._plot_offset)
            self.curve.append(self.c)

        # Initialise buffer for the plotter (maximum of 10 seconds)
        self._buffer_size = 10  # seconds
        self.window_buffer = (
            np.ones(
                (
                    self.active_channels,
                    int(np.ceil(self.sample_rate * self._buffer_size)),
                )
            )
            * np.nan
        )
        self.samples_seen = 0

    def init_impedance_table(self):
        self.gui_handle.table_live_impedance.setColumnCount(2)
        self.gui_handle.table_live_impedance.setHorizontalHeaderLabels(
            ["Name", "Impedance"]
        )
        self.gui_handle.table_live_impedance.setRowCount(len(self.chs) - 5)

        for i, channel in enumerate(self.chs):
            if i < len(self.chs) - 5:
                alt_name = QtWidgets.QTableWidgetItem(
                    self.chs[self.channel_conversion_list[i]][0]
                )
                alt_name.setFlags(Qt.ItemIsEnabled)
                real = QtWidgets.QTableWidgetItem("kOhm")
                real.setFlags(Qt.ItemIsEnabled)
                self.gui_handle.table_live_impedance.setItem(i, 0, alt_name)
                self.gui_handle.table_live_impedance.setItem(i, 1, real)

    def _decrease_time_range(self):
        """Method that decreases the amount of time that is displayed within the
        window.
        """
        # Minimum window size is 1 second
        if self.window_size > 1:
            self.window_size -= 1
            self.worker.window_size -= 1

            self.RealTimePlotWidget.window.setXRange(
                0.02 * self.window_size, 0.97 * self.window_size
            )

            self.gui_handle.increase_time_button.setEnabled(True)
            self.gui_handle.increase_time_button.setText(
                "Increase time range: " + str(self.window_size + 1) + "s"
            )
            if self.window_size == 1:
                self.gui_handle.decrease_time_button.setEnabled(False)
            else:
                self.gui_handle.decrease_time_button.setEnabled(True)
                self.gui_handle.decrease_time_button.setText(
                    "Decrease time range: " + str(self.window_size - 1) + "s"
                )

    def _increase_time_range(self):
        """Method that increases the amount of time that is displayed within the
        window.
        """
        # Maximum window size is defined by the _buffer_size parameter
        if self.window_size < self._buffer_size:
            self.window_size += 1
            self.worker.window_size += 1

            self.RealTimePlotWidget.window.setXRange(
                0.02 * self.window_size, 0.97 * self.window_size
            )

            self.gui_handle.decrease_time_button.setEnabled(True)
            self.gui_handle.decrease_time_button.setText(
                "Decrease time range: " + str(self.window_size - 1) + "s"
            )
            if self.window_size == self._buffer_size:
                self.gui_handle.increase_time_button.setEnabled(False)
            else:
                self.gui_handle.increase_time_button.setEnabled(True)
                self.gui_handle.increase_time_button.setText(
                    "Increase time range: " + str(self.window_size + 1) + "s"
                )

    def _update_scale(self, type_flag):
        """Method responsible for updating the scale whenever user input is
        provided to do so.
        """
        # Find white-out region using the Status channel (can't be 0)
        x, y = self.curve[-1].getData()
        # Transformed data is plotted so convert back to original values for correct
        # display of the values on the y-axis
        y = y * -1 * self._plot_diff[-1]["diff"] + self._plot_diff[-1]["mean"]

        # The white-out region is entered in the plot as zeros, which have to be
        # omitted from the scaling calculation
        idx_data = np.ones(np.size(y), dtype=bool)
        idx_whiteout = np.where(np.abs(y) == 0)[0]
        idx_data[idx_whiteout] = False

        # Loop over all channels to find individual scaling factors
        for i in range(self.num_channels):
            x, y = self.curve[i].getData()

            # Convert to original sample data
            y = y * -1 * self._plot_diff[i]["diff"] + self._plot_diff[i]["mean"]

            # Find the mean and the difference between minimum and maximum
            self._plot_diff[i]["mean"] = (
                np.max(y[idx_data]) + np.min(y[idx_data])
            ) / 2
            if type_flag == "scale":
                self._plot_diff[i]["diff"] = np.abs(
                    np.max(y[idx_data]) - np.min(y[idx_data])
                )
            elif type_flag == "range":
                self._plot_diff[i]["diff"] = int(
                    self.gui_handle.set_range_box.currentText()
                )

            # Whenever there is no difference, the difference is reset to 2^31
            if self._plot_diff[i]["diff"] == 0:
                self._plot_diff[i]["diff"] = 2**31
                if i == self.num_channels - 1:
                    self._plot_diff[i]["diff"] = 1

        # Update the range of the displayed y-axis range
        self.RealTimePlotWidget.window.setYRange(
            -self._plot_offset / 3,
            (self.num_channels - 2 / 3) * self._plot_offset,
        )

        # Update the list of ticks on the y-axis for all channels
        tick_list_left = [[]]

        if isinstance(self.device, TMSiDevice):
            for i in range(self.num_channels):
                for j in [-1, 0, 1]:
                    if i == self.num_channels - 1:
                        if not bool(j):
                            tick_list_left[0].append(
                                (
                                    int(self._plot_offset * i),
                                    f"{self.chs[self.gui_handle._channel_selection[i]][0]: <25}",
                                )
                            )
                        else:
                            tick_list_left[0].append(
                                (
                                    int(
                                        self._plot_offset * i
                                        + -j * self._plot_offset / 3
                                    ),
                                    f'{ self._plot_diff[i]["mean"] + (self._plot_diff[i]["diff"] * j ) : .2g}',
                                )
                            )
                    else:
                        if not bool(j):
                            tick_list_left[0].append(
                                (
                                    int(self._plot_offset * i),
                                    f"{self.chs[self.channel_conversion_list[self.gui_handle._channel_selection[i]]][0]: <25}",
                                )
                            )
                        else:
                            tick_list_left[0].append(
                                (
                                    int(
                                        self._plot_offset * i
                                        + -j * self._plot_offset / 3
                                    ),
                                    f'{ self._plot_diff[i]["mean"] + (self._plot_diff[i]["diff"] * j ) : .2g}',
                                )
                            )
        else:
            for i in range(self.num_channels):
                for j in [-1, 0, 1]:
                    if i == self.num_channels - 1:
                        if not bool(j):
                            tick_list_left[0].append(
                                (
                                    int(self._plot_offset * i),
                                    f"{self.device.channels[self.gui_handle._channel_selection[i]].name: <25}",
                                )
                            )
                        else:
                            tick_list_left[0].append(
                                (
                                    int(
                                        self._plot_offset * i
                                        + -j * self._plot_offset / 3
                                    ),
                                    f'{ self._plot_diff[i]["mean"] + (self._plot_diff[i]["diff"] * j ) : .2g}',
                                )
                            )
                    else:
                        if not bool(j):
                            tick_list_left[0].append(
                                (
                                    int(self._plot_offset * i),
                                    f"{self.device.channels[self.channel_conversion_list[self.gui_handle._channel_selection[i]]].name: <25}",
                                )
                            )
                        else:
                            tick_list_left[0].append(
                                (
                                    int(
                                        self._plot_offset * i
                                        + -j * self._plot_offset / 3
                                    ),
                                    f'{ self._plot_diff[i]["mean"] + (self._plot_diff[i]["diff"] * j ) : .2g}',
                                )
                            )

        self.RealTimePlotWidget.window.getAxis("left").setTicks(tick_list_left)

    def _update_channel_display(self):
        """Method that updates which channels are displayed."""

        # Make a copy of the current channel indices that are displayed
        _idx_curve_list = np.copy(self.gui_handle._channel_selection)

        # Update the channel selection based on the clicked checkboxes
        for i in range(np.size(self.gui_handle._checkboxes, 0)):
            if self.gui_handle._checkboxes[i][0].isChecked():
                if i in self.gui_handle._channel_selection:
                    pass
                else:
                    # Create a flag that states if the number of channels displayed increases
                    _increase = True
                    self.gui_handle._channel_selection = np.hstack(
                        (self.gui_handle._channel_selection, i)
                    )
            else:
                if i in self.gui_handle._channel_selection:
                    # Create a flag that states if the number of channels displayed decreases
                    _increase = False
                    self.gui_handle._channel_selection = np.delete(
                        self.gui_handle._channel_selection,
                        self.gui_handle._channel_selection == i,
                    )
                else:
                    pass

        # Sort the indices from small to large, to update them appropriately in the plot
        self.gui_handle._channel_selection.sort()

        # Update the num_channels parameter that keeps track of the amount of channels that are displayed in the plot
        self.num_channels = np.size(self.gui_handle._channel_selection, 0)

        # Make a copy of the dictionary containing the scaling parameters of all channels
        copy_plot_diff = self._plot_diff.copy()

        # Initialise a counter parameter required for passing the scaling factors
        count = 0

        # Create a new dictionary with scaling parameters, which is updated with the old values in the next step
        self._plot_diff = [
            {"mean": 0, "diff": 2**31} for i in range(self.num_channels)
        ]

        # Depending on whether the displayed number of channels increases or decreases, pass the old scaling factors appropriately
        if _increase:
            _idx_overlap = [
                np.where(i == self.gui_handle._channel_selection)[0][0]
                for idx, i in enumerate(_idx_curve_list)
                if i in self.gui_handle._channel_selection
            ]

            for i in range(self.num_channels):
                if i in _idx_overlap:
                    self._plot_diff[i]["mean"] = copy_plot_diff[count]["mean"]
                    self._plot_diff[i]["diff"] = copy_plot_diff[count]["diff"]
                    count += 1
        else:
            _idx_overlap = [
                np.where(i == _idx_curve_list)[0][0]
                for idx, i in enumerate(_idx_curve_list)
                if i in self.gui_handle._channel_selection
            ]

            for i in range(self.num_channels):
                self._plot_diff[i]["mean"] = copy_plot_diff[
                    _idx_overlap[count]
                ]["mean"]
                self._plot_diff[i]["diff"] = copy_plot_diff[
                    _idx_overlap[count]
                ]["diff"]
                count += 1

        # Reset the parameter keeping track of the plotted graphis items and clear the window
        self.curve = []
        self.RealTimePlotWidget.window.clear()

        # Generate the new graphics objects required for plotting
        for i in range(self.num_channels):
            self.c = pg.PlotCurveItem(pen="black")
            self.RealTimePlotWidget.window.addItem(self.c)
            self.c.setPos(0, (i) * self._plot_offset)
            self.curve.append(self.c)

        # Update the range of the displayed y-axis range so that all channels are included in the plot
        self.RealTimePlotWidget.window.setYRange(
            -self._plot_offset / 3,
            (self.num_channels - 2 / 3) * self._plot_offset,
        )

        # Update the list of ticks on the y-axis for all channels
        tick_list_left = [[]]

        if isinstance(self.device, TMSiDevice):
            for i in range(self.num_channels):
                for j in [-1, 0, 1]:
                    if i == self.num_channels - 1:
                        if not bool(j):
                            tick_list_left[0].append(
                                (
                                    int(self._plot_offset * i),
                                    f"{self.chs[self.gui_handle._channel_selection[i]][0]: <25}",
                                )
                            )
                        else:
                            tick_list_left[0].append(
                                (
                                    int(
                                        self._plot_offset * i
                                        + -j * self._plot_offset / 3
                                    ),
                                    f'{ self._plot_diff[i]["mean"] + (self._plot_diff[i]["diff"] * j ) : .2g}',
                                )
                            )
                    else:
                        if not bool(j):
                            tick_list_left[0].append(
                                (
                                    int(self._plot_offset * i),
                                    f"{self.chs[self.channel_conversion_list[self.gui_handle._channel_selection[i]]][0]: <25}",
                                )
                            )
                        else:
                            tick_list_left[0].append(
                                (
                                    int(
                                        self._plot_offset * i
                                        + -j * self._plot_offset / 3
                                    ),
                                    f'{ self._plot_diff[i]["mean"] + (self._plot_diff[i]["diff"] * j ) : .2g}',
                                )
                            )
        else:
            for i in range(self.num_channels):
                for j in [-1, 0, 1]:
                    if i == self.num_channels - 1:
                        if not bool(j):
                            tick_list_left[0].append(
                                (
                                    int(self._plot_offset * i),
                                    f"{self.device.channels[self.gui_handle._channel_selection[i]].name: <25}",
                                )
                            )
                        else:
                            tick_list_left[0].append(
                                (
                                    int(
                                        self._plot_offset * i
                                        + -j * self._plot_offset / 3
                                    ),
                                    f'{ self._plot_diff[i]["mean"] + (self._plot_diff[i]["diff"] * j ) : .2g}',
                                )
                            )
                    else:
                        if not bool(j):
                            tick_list_left[0].append(
                                (
                                    int(self._plot_offset * i),
                                    f"{self.device.channels[self.channel_conversion_list[self.gui_handle._channel_selection[i]]].name: <25}",
                                )
                            )
                        else:
                            tick_list_left[0].append(
                                (
                                    int(
                                        self._plot_offset * i
                                        + -j * self._plot_offset / 3
                                    ),
                                    f'{ self._plot_diff[i]["mean"] + (self._plot_diff[i]["diff"] * j ) : .2g}',
                                )
                            )

        self.RealTimePlotWidget.window.getAxis("left").setTicks(tick_list_left)

    def _show_all_UNI(self):
        """Method that checks all channels of type UNI so that they are displayed
        in the plot
        """

        # Find all channel indices that are of type UNI
        UNI_lst = [
            i
            for i in range(np.size(self.gui_handle._checkboxes, 0))
            if self.gui_handle._checkboxes[i][1].value == ChannelType.UNI.value
        ]

        # When all UNI channels are already checked, do not update the plot
        if all(self.gui_handle._checkboxes[i][0].isChecked() for i in UNI_lst):
            return

        # Set all UNI channels' checkboxes to the Checked state
        for i in range(np.size(self.gui_handle._checkboxes, 0)):
            if self.gui_handle._checkboxes[i][1].value == ChannelType.UNI.value:
                self.gui_handle._checkboxes[i][0].setChecked(True)

        # Call the update function (normally called by individual checkboxes)
        self._update_channel_display()

    def _hide_all_UNI(self):
        """Method that unchecks all channels of type UNI so that they are not
        displayed in the plot
        """

        # Find all channel indices that are of type UNI
        UNI_lst = [
            i
            for i in range(np.size(self.gui_handle._checkboxes, 0))
            if self.gui_handle._checkboxes[i][1].value == ChannelType.UNI.value
        ]

        # When all UNI channels are already unchecked, do not update the plot
        if all(
            not self.gui_handle._checkboxes[i][0].isChecked() for i in UNI_lst
        ):
            return

        # Set all UNI channels' checkboxes to the Checked state
        for i in range(np.size(self.gui_handle._checkboxes, 0)):
            if self.gui_handle._checkboxes[i][1].value == ChannelType.UNI.value:
                self.gui_handle._checkboxes[i][0].setChecked(False)

        # Call the update function (normally called by individual checkboxes)
        self._update_channel_display()

    def _show_all_BIP(self):
        """Method that checks all channels of type BIP so that they are displayed
        in the plot
        """

        # Find all channel indices that are of type BIP
        BIP_lst = [
            i
            for i in range(np.size(self.gui_handle._checkboxes, 0))
            if self.gui_handle._checkboxes[i][1].value == ChannelType.BIP.value
        ]

        # When all BIP channels are already checked, do not update the plot
        if all(self.gui_handle._checkboxes[i][0].isChecked() for i in BIP_lst):
            return

        # Set all BIP channels' checkboxes to the Checked state
        for i in range(np.size(self.gui_handle._checkboxes, 0)):
            if self.gui_handle._checkboxes[i][1].value == ChannelType.BIP.value:
                self.gui_handle._checkboxes[i][0].setChecked(True)

        # Call the update function (normally called by individual checkboxes)
        self._update_channel_display()

    def _hide_all_BIP(self):
        """Method that unchecks all channels of type BIP so that they are not
        displayed in the plot
        """

        # Find all channel indices that are of type BIP
        BIP_lst = [
            i
            for i in range(np.size(self.gui_handle._checkboxes, 0))
            if self.gui_handle._checkboxes[i][1].value == ChannelType.BIP.value
        ]

        # When all BIP channels are already unchecked, do not update the plot
        if all(
            not self.gui_handle._checkboxes[i][0].isChecked() for i in BIP_lst
        ):
            return

        # Set all BIP channels' checkboxes to the Checked state
        for i in range(np.size(self.gui_handle._checkboxes, 0)):
            if self.gui_handle._checkboxes[i][1].value == ChannelType.BIP.value:
                self.gui_handle._checkboxes[i][0].setChecked(False)

        # Call the update function (normally called by individual checkboxes)
        self._update_channel_display()

    def _show_all_AUX(self):
        """Method that checks all channels of type AUX so that they are displayed
        in the plot
        """

        # Find all channel indices that are of type AUX
        AUX_lst = [
            i
            for i in range(np.size(self.gui_handle._checkboxes, 0))
            if self.gui_handle._checkboxes[i][1].value == ChannelType.AUX.value
        ]

        # When all AUX channels are already checked, do not update the plot
        if all(self.gui_handle._checkboxes[i][0].isChecked() for i in AUX_lst):
            return

        # Set all AUX channels' checkboxes to the Checked state
        for i in range(np.size(self.gui_handle._checkboxes, 0)):
            if self.gui_handle._checkboxes[i][1].value == ChannelType.AUX.value:
                self.gui_handle._checkboxes[i][0].setChecked(True)

        # Call the update function (normally called by individual checkboxes)
        self._update_channel_display()

    def _hide_all_AUX(self):
        """Method that unchecks all channels of type AUX so that they are not
        displayed in the plot
        """

        # Find all channel indices that are of type AUX
        AUX_lst = [
            i
            for i in range(np.size(self.gui_handle._checkboxes, 0))
            if self.gui_handle._checkboxes[i][1].value == ChannelType.AUX.value
        ]

        # When all AUX channels are already unchecked, do not update the plot
        if all(
            not self.gui_handle._checkboxes[i][0].isChecked() for i in AUX_lst
        ):
            return

        # Set all AUX channels' checkboxes to the Checked state
        for i in range(np.size(self.gui_handle._checkboxes, 0)):
            if self.gui_handle._checkboxes[i][1].value == ChannelType.AUX.value:
                self.gui_handle._checkboxes[i][0].setChecked(False)

        # Call the update function (normally called by individual checkboxes)
        self._update_channel_display()

    def _show_all_DIGI(self):
        """Method that checks all channels of type sensor so that they are displayed
        in the plot
        """

        # Find all channel indices that are of type sensor
        DIGI_lst = [
            i
            for i in range(np.size(self.gui_handle._checkboxes, 0))
            if self.gui_handle._checkboxes[i][1].value
            == ChannelType.sensor.value
        ]

        # When all sensor channels are already checked, do not update the plot
        if all(self.gui_handle._checkboxes[i][0].isChecked() for i in DIGI_lst):
            return

        # Set all sensor channels' checkboxes to the Checked state
        for i in range(np.size(self.gui_handle._checkboxes, 0)):
            if (
                self.gui_handle._checkboxes[i][1].value
                == ChannelType.sensor.value
            ):
                self.gui_handle._checkboxes[i][0].setChecked(True)

        # Call the update function (normally called by individual checkboxes)
        self._update_channel_display()

    def _hide_all_DIGI(self):
        """Method that unchecks all channels of type sensor so that they are not
        displayed in the plot
        """

        # Find all channel indices that are of type DIGI
        DIGI_lst = [
            i
            for i in range(np.size(self.gui_handle._checkboxes, 0))
            if self.gui_handle._checkboxes[i][1].value
            == ChannelType.sensor.value
        ]

        # When all DIGI channels are already unchecked, do not update the plot
        if all(
            not self.gui_handle._checkboxes[i][0].isChecked() for i in DIGI_lst
        ):
            return

        # Set all DIGI channels' checkboxes to the Checked state
        for i in range(np.size(self.gui_handle._checkboxes, 0)):
            if (
                self.gui_handle._checkboxes[i][1].value
                == ChannelType.sensor.value
            ):
                self.gui_handle._checkboxes[i][0].setChecked(False)

        # Call the update function (normally called by individual checkboxes)
        self._update_channel_display()

    @QtCore.Slot(object)
    def update_plot(self, data):
        """Method that receives the data from the sampling thread and writes
        it to the GUI window.
        """

        # PyQtGraph can't handle NaN-values, therefore the WhiteOut has to be implemented differently.
        # This is done using a boolean array that states which points should not be connected (NaN values)
        con = np.isfinite(data)
        data[~con] = 0

        # Find final index for updating values in right-side yticks list
        idx_final = np.where(con[0, :] == False)
        # Check whether the vector is not empty (can occur upon rescaling) and
        # stop the plot update when this is the case
        if not idx_final[0].any():
            return

        # The last added value is the value before the White-out region.
        # As the white-out region resets each time the end of the window is reached,
        # different logic is needed when the white-out region lies around the wrapping points of the window
        if idx_final[0][0] != 0:
            idx_final = idx_final[0][0] - 1
        elif (
            idx_final[0][0] == 0
            and idx_final[0][-1]
            != (self.window_size * self.sample_rate) / self._downsampling_factor
            - 1
        ):
            idx_final = (
                self.window_size * self.sample_rate
            ) / self._downsampling_factor - 1
        else:
            dummy_idx = np.where(
                idx_final[0][:]
                > 0.9
                * (self.window_size * self.sample_rate)
                / self._downsampling_factor
            )
            idx_final = idx_final[0][dummy_idx[0][0]] - 1

        # Update the x-axis ticks so that the time base is reflected correctly on the x-axis
        t_end = int(np.nanmax(data[-1, :] / self.sample_rate))
        bottom_ticks = [
            [
                (val % self.window_size, str(val))
                for val in np.arange(
                    t_end - (self.window_size - 1), t_end + 1, dtype=int
                )
            ]
        ]
        self.RealTimePlotWidget.window.getAxis("bottom").setTicks(bottom_ticks)

        # Ensure right amount of data points are plotted (window_size * sample_rate)
        time_axis = np.arange(
            0, self.window_size, self._downsampling_factor / self.sample_rate
        )

        # Try to update the plot, due to user actions plotting might result in a warning
        # for that specific plot instance, hence the try-except statement
        try:
            for i in range(self.num_channels):
                # Draw data (apply scaling and multiply with negative 1 (needed due to inverted y axis))
                self.curve[i].setData(
                    time_axis,
                    (
                        data[self.gui_handle._channel_selection[i], :]
                        - self._plot_diff[i]["mean"]
                    )
                    / self._plot_diff[i]["diff"]
                    * -1,
                    connect=np.logical_and(con[i, :], np.roll(con[i, :], -1)),
                )

            # Update the ticks on the right side of the plot
            if isinstance(self.device, TMSiDevice):
                tick_list_right = [
                    [
                        (
                            int(self._plot_offset * i),
                            f"{data[self.gui_handle._channel_selection[i],int(idx_final)]:< 10.2f} {self.chs[self.gui_handle._channel_selection[i]][1]}",
                        )
                        for i in range(self.num_channels)
                    ]
                ]
            else:
                tick_list_right = [
                    [
                        (
                            int(self._plot_offset * i),
                            f"{data[self.gui_handle._channel_selection[i],int(idx_final)]:< 10.2f} {self.device.channels[self.gui_handle._channel_selection[i]].unit_name}",
                        )
                        for i in range(self.num_channels)
                    ]
                ]
            self.RealTimePlotWidget.window.getAxis("right").setTicks(
                tick_list_right
            )
        except Exception:
            pass

    @QtCore.Slot(object)
    def update_impedance_table(self, live_impedances):
        for i, channel in enumerate(self.chs):
            if i < len(self.chs) - 5:
                real = QtWidgets.QTableWidgetItem(
                    f'{live_impedances[i+1]["Re"]} kOhm'
                )
                real.setFlags(Qt.ItemIsEnabled)
                self.gui_handle.table_live_impedance.setItem(i, 1, real)

    def setupThread(self):
        """Method that initialises the sampling thread of the device"""

        # Create a Thread
        self.thread = QtCore.QThread()
        # Instantiate the Sampling class
        self.worker = SamplingThread(self)

        # Move the worker to a Thread
        self.worker.moveToThread(self.thread)

        # Connect signals to slots depending on whether the plotter is active
        if self.filter_app:
            self.thread.started.connect(self.worker.update_filtered_samples)
        else:
            self.thread.started.connect(self.worker.update_samples)
        self.worker.output.connect(self.update_plot)
        self.worker.output_impedance_table.connect(self.update_impedance_table)


class SamplingThread(QtCore.QObject):
    """Class responsible for sampling and preparing data for the GUI window."""

    # Initialise the ouptut object
    output = QtCore.Signal(object)
    output_impedance_table = QtCore.Signal(object)

    def __init__(self, main_class):
        """Setting up the class' properties that were passed from the GUI thread"""
        QtCore.QObject.__init__(self)
        # Access initialised values from the GUI class
        self.num_channels = main_class.num_channels
        self.sample_rate = main_class.sample_rate
        self.window_size = main_class.window_size
        self.window_buffer = main_class.window_buffer
        self._buffer_size = main_class._buffer_size
        self.samples_seen = main_class.samples_seen
        self.device = main_class.device
        self._downsampling_factor = main_class._downsampling_factor
        self.filter_app = main_class.filter_app
        self.channel_conversion_list = main_class.channel_conversion_list
        self.grid_type = main_class.grid_type
        self.live_impedance = main_class.live_impedance
        if self.live_impedance:
            self._cycling_impedance = dict()
            for index in range(1, len(main_class.chs) - 4):
                self._cycling_impedance[index] = dict()
                self._cycling_impedance[index]["Re"] = 1000
                self._cycling_impedance[index]["Im"] = 1000

        # Register to filter_app or sample data server and start measurement
        if not self.filter_app:
            # Prepare Queue
            _QUEUE_SIZE = 1000
            self.q_sample_sets = queue.Queue(_QUEUE_SIZE)
            if isinstance(self.device, TMSiDevice):
                ApexSampleDataServer().register_consumer(
                    self.device.get_id(), self.q_sample_sets
                )
            else:
                # Register the consumer to the sample data server
                sample_data_server.registerConsumer(
                    self.device.id, self.q_sample_sets
                )

        # Set sampling to true
        self.sampling = True

    @QtCore.Slot()
    def update_samples(self):
        """Method that retrieves samples from the queue and processes the samples.
        Processing includes reshaping the data into the desired format, adding
        white out region to show the refresh rate of the plot.
        """
        lag = False
        while self.sampling:
            while not self.q_sample_sets.empty():
                # If the sample_data_server queue contains more than 10 items, the queue is trimmed so that the plotter
                # has a lower delay
                if self.q_sample_sets.qsize() > 10:
                    lag = True
                    print(
                        "The plotter skipped some samples to compensate for lag.."
                    )

                elif self.q_sample_sets.qsize() < 6:
                    lag = False

                # Retrieve sample data from the sample_data_server queue
                sd = self.q_sample_sets.get()
                self.q_sample_sets.task_done()

                # Reshape the samples retrieved from the queue
                samples = np.reshape(
                    sd.samples,
                    (sd.num_samples_per_sample_set, sd.num_sample_sets),
                    order="F",
                )

                # Add a White out region to show the update of the samples
                white_out = int(
                    np.floor(self.window_size * self.sample_rate * 0.04)
                )
                plot_indices = (
                    self.samples_seen
                    + (np.arange(np.size(samples, 1) + white_out))
                ) % (self.sample_rate * self._buffer_size)

                # Write sample data to the buffer
                self.window_buffer[:, plot_indices[:-white_out]] = samples

                # Apply the conversion strategy
                self.window_buffer[:, plot_indices[:-white_out]] = samples[
                    self.channel_conversion_list, :
                ]

                self.window_buffer[:, plot_indices[-white_out:]] = np.nan

                # Update number of samples seen by the plotter
                self.samples_seen += np.size(samples, 1)

                # When the plotter lags, don't output the sample data to the plotter until (most of) the lag is gone
                if lag:
                    time.sleep(0.001)
                else:

                    # The indices to be plotted ranges until the final index of the white out region,
                    # and has a total number of samples equal to the window size
                    indices = np.arange(
                        (
                            (self.samples_seen + white_out)
                            - (self.window_size * self.sample_rate)
                        ),
                        (self.samples_seen + white_out),
                        dtype=int,
                    )

                    # The indices have to match with the indices of the window buffer
                    scroll_idx = (indices) % (
                        self.sample_rate * self._buffer_size
                    )

                    # The buffer indices have a wrapping point (e.g. in a buffer of 4000 samples, the wrapping point is from 3999 to 0)
                    # As the window size might not line up with this exactly,
                    # the wrapping point needs to be identified to ensure that the plot does not 'jump'.
                    split_idx = np.where(
                        indices % (self.sample_rate * self.window_size) == 1
                    )[0][0]

                    # Retrieve the correct sample data that needs to be plotted
                    plot_data = np.hstack(
                        (
                            self.window_buffer[:, scroll_idx[split_idx:]],
                            self.window_buffer[:, scroll_idx[0:split_idx]],
                        )
                    )

                    plot_data = plot_data[:, :: self._downsampling_factor]

                    # Output sample data
                    self.output.emit(plot_data)

                    # Update live impedance data
                    if self.live_impedance:
                        self.update_impedances(samples)
                        self.output_impedance_table.emit(
                            self._cycling_impedance
                        )

                    # Pause the thread for a small time so that plot can be updated before receiving next data chunk
                    # Pause should be long enough to have the screen update itself
                    time.sleep(0.03)

    @QtCore.Slot()
    def update_filtered_samples(self):
        """Method that retrieves samples from the queue and processes the samples.
        Processing includes reshaping the data into the desired format, adding
        white out region to show the refresh rate of the plot.
        """
        lag = False
        while self.sampling:
            while not self.filter_app.q_filtered_sample_sets.empty():
                if self.filter_app.q_filtered_sample_sets.qsize() > 10:
                    lag = True
                    print(
                        "The plotter skipped some samples to compensate for lag.."
                    )

                elif self.filter_app.q_filtered_sample_sets.qsize() < 6:
                    lag = False

                sd = self.filter_app.q_filtered_sample_sets.get()
                self.filter_app.q_filtered_sample_sets.task_done()
                samples = copy(sd)

                # Add a White out region to show the update of the samples
                white_out = int(
                    np.floor(self.window_size * self.sample_rate * 0.04)
                )
                plot_indices = (
                    self.samples_seen
                    + (np.arange(np.size(samples, 1) + white_out))
                ) % (self.sample_rate * self._buffer_size)

                # Write sample data to the plot buffer
                self.window_buffer[:, plot_indices[:-white_out]] = samples

                # Apply the conversion strategy
                self.window_buffer[:, plot_indices[:-white_out]] = samples[
                    self.channel_conversion_list, :
                ]

                self.window_buffer[:, plot_indices[-white_out:]] = np.nan

                # Update number of samples seen by the plotter
                self.samples_seen += np.size(samples, 1)

                if lag:
                    time.sleep(0.001)
                else:
                    # The indices to be plotted ranges until the final index of the white out region,
                    # and has a total number of samples equal to the window size
                    indices = np.arange(
                        (
                            (self.samples_seen + white_out)
                            - (self.window_size * self.sample_rate)
                        ),
                        (self.samples_seen + white_out),
                        dtype=int,
                    )

                    # The indices have to match with the indices of the window buffer
                    scroll_idx = (indices) % (
                        self.sample_rate * self._buffer_size
                    )

                    # The buffer indices have a wrapping point (e.g. in a buffer of 4000 samples, the wrapping point is from 3999 to 0)
                    # As the window size might not line up with this exactly,
                    # the wrapping point needs to be identified to ensure that the plot does not 'jump'.
                    split_idx = np.where(
                        indices % (self.sample_rate * self.window_size) == 1
                    )[0][0]

                    # Retrieve the correct sample data that needs to be plotted
                    plot_data = np.hstack(
                        (
                            self.window_buffer[:, scroll_idx[split_idx:]],
                            self.window_buffer[:, scroll_idx[0:split_idx]],
                        )
                    )

                    plot_data = plot_data[:, :: self._downsampling_factor]

                    # Output sample data
                    self.output.emit(plot_data)

                    # Update live impedance data
                    if self.live_impedance:
                        self.update_impedances(samples)
                        self.output_impedance_table.emit(
                            self._cycling_impedance
                        )

                    # Pause the thread for a small time so that plot can be updated before receiving next data chunk
                    # Pause should be long enough to have the screen update itself
                    time.sleep(0.03)

    def update_impedances(self, samples):
        CYCL_IDX = len(samples[:, 0]) - 5
        for idx in range(len(samples[CYCL_IDX, :])):
            index = int(samples[CYCL_IDX, idx]) + 1
            if index in self._cycling_impedance:
                self._cycling_impedance[index]["Re"] = samples[
                    CYCL_IDX + 1, idx
                ]
                self._cycling_impedance[index]["Im"] = samples[
                    CYCL_IDX + 2, idx
                ]
            else:
                self._cycling_impedance[index] = dict()
                self._cycling_impedance[index]["Re"] = samples[
                    CYCL_IDX + 1, idx
                ]
                self._cycling_impedance[index]["Im"] = samples[
                    CYCL_IDX + 2, idx
                ]

    def stop(self):
        """Method that is executed when the thread is terminated.
        This stop event stops the measurement.
        """
        self.sampling = False


if __name__ == "__main__":
    # Initialise the TMSi-SDK first before starting using it
    tmsi_device.initialize()

    # Create the device object to interface with the SAGA-system.
    dev = tmsi_device.create(
        tmsi_device.DeviceType.saga,
        DeviceInterfaceType.docked,
        DeviceInterfaceType.network,
    )

    # Find and open a connection to the SAGA-system and print its serial number
    dev.open()
    print("handle 1 " + str(dev.info.ds_serial_number))

    # Initialise the application
    app = QtGui.QApplication(sys.argv)
    # Define the GUI object and show it
    window = RealTimePlot(figurename="A RealTimePlot", device=dev)
    window.show()

    # Enter the event loop
    app.exec_()

    dev.close()
