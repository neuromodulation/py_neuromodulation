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
 * @file ${hd_emg_plotter.py} 
 * @brief Plotter object that displays an heatmap based on real-time computed
 * RMS values, designed for HD-EMG applications.
 *
 */


'''

from PySide2 import QtGui, QtCore, QtWidgets
import numpy as np
import pyqtgraph as pg
import time
import queue
from scipy import signal, interpolate
import sys


from os.path import join, dirname, realpath, normpath

Plotter_dir = dirname(realpath(__file__)) # directory of this file
measurements_dir = join(Plotter_dir, '../../measurements') # directory with all measurements
modules_dir = normpath(join(Plotter_dir, '../..')) # directory with all modules

from TMSiSDK import tmsi_device
from TMSiSDK import sample_data_server

from TMSiSDK.device import DeviceInterfaceType, ChannelType


class HeatMapViewer():
    """ Class that creates a GUI to display the impedance values in a gridded
        layout.
    """
    def __init__(self, gui_handle, device, tail_orientation, signal_lim, grid_type = 'flex pcb grid'):
        """ Setting up the GUI's elements
        """
        
        if sys.platform == "linux" or sys.platform == "linux2":
            print('This plotter is not compatible with the current version of the TMSi Python Interface on Linux (Ubuntu 18.04 LTS)\n')
            return
        
        # Pass the device handle to the GUI
        self.device = device 
        
        self.gui_handle = gui_handle
        self.RealTimePlotWidget = self.gui_handle.RealTimePlotWidget
        
        # Determine used number of channels
        if self.device.config.num_channels > 64: 
            if '4-8' in grid_type or grid_type[-1] == '1' or grid_type[-1] == '2':
                self._EMG_chans = 32
            else:
                self._EMG_chans = 64
        else:
            self._EMG_chans = 32
        
        self._preprocess_wifi = False
        if self.device.info.dr_interface == DeviceInterfaceType.wifi:
            self._preprocess_wifi = True
        
        self.sample_rate = self.device.config.get_sample_rate(ChannelType.counter)
        
        # Orientation of the grid on the body
        self.tail_orientation = tail_orientation
        
        # Upper limit colorbar
        self.signal_lim = signal_lim
        
        # Type of used grid
        self.grid_type = grid_type
        if not '11' in self.grid_type:
            self.n_x = int(self._EMG_chans/8)
            self.n_y = 8
        else:
            self.n_x = int(np.ceil(self._EMG_chans/11))
            self.n_y = 11
        
        # Set up UI and thread
        self.initUI()
        self.setupThread()
        
    def initUI(self):
        """ Method responsible for constructing the basic elements in the plot.
            All viewboxes have a set size so that the information can be displayed
            correctly.
        """
        # Set view settings
        self.RealTimePlotWidget.setBackground('w')
        # self.showMaximized()
        
        # Add plot window for the channels
        self.RealTimePlotWidget.window = self.RealTimePlotWidget.addPlot()
        self.RealTimePlotWidget.window.setAspectLocked(lock=True, ratio = 1)
        
        # Write the ticks to the plot
        self.RealTimePlotWidget.window.hideAxis('left')
        self.RealTimePlotWidget.window.hideAxis('bottom')
        
        # Disable auto-scaling and menu
        self.RealTimePlotWidget.window.hideButtons()
        self.RealTimePlotWidget.window.setMenuEnabled(False)
        self.RealTimePlotWidget.window.setMouseEnabled(x = False, y = False)
        
        # Delete CREF channel from channel conversion list
        self.channel_conversion_list = np.delete(self.gui_handle.channel_conversion_list[:self._EMG_chans+1], 0) 
        
        # Detect disabled channels
        self.channel_insertion_list = []
        offset=0
        for ch in range(self._EMG_chans+1):
            if not self.device.channels[ch-offset].name == self.device._config._channels[ch].alt_name:
                offset  =offset + 1
                self.channel_insertion_list.append(ch)
        
        # Insert dummy channels
        self.dummy_chan = []
        if '11' in self.grid_type:
            if not self.grid_type[-1] == '2':
                self.dummy_chan.append(10)
                self.channel_conversion_list = np.insert(self.channel_conversion_list, 10, max(self.channel_conversion_list)+1)
            if not self.grid_type[-1] == '1':
                dummy_chan=max(self.channel_conversion_list)+1
                self.dummy_chan.append(dummy_chan)
                self.channel_conversion_list = np.append(self.channel_conversion_list, dummy_chan)        
        
        # Ratio is slightly different between 32/64 channel setup
        # Number of rows/columns depends on grid-type
        if not '11' in self.grid_type:
            if self._EMG_chans == 32:
                self._x_interpolate = np.arange(0, int(self._EMG_chans/8) - 1 + .1, 0.1)
                self._y_interpolate = np.arange(0, int(self._EMG_chans/(self._EMG_chans/8)) - 1 + .1, 0.1)
            else:
                self._x_interpolate = np.arange(0, int(self._EMG_chans/8) - 1 + .2, 0.1)
                self._y_interpolate = np.arange(0, int(self._EMG_chans/(self._EMG_chans/8)) - 1 + .1, 0.1)
        else:
            self._x_interpolate = np.arange(0, self.n_x - 1 + .1, 0.1)
            self._y_interpolate = np.arange(0, self.n_y - 1 + .1, 0.1)
        self._dummy_val = np.zeros((len(self._x_interpolate), len(self._y_interpolate)))
        
        if self.tail_orientation == 'Right' or self.tail_orientation == 'right':
            self._dummy_val = np.rot90(self._dummy_val, 1)
        elif self.tail_orientation == 'Left' or self.tail_orientation == 'left':
            self._dummy_val = np.rot90(self._dummy_val, 3)
            
        # Create image item
        self.img = pg.ImageItem(image = self._dummy_val) 
        self.RealTimePlotWidget.window.addItem(self.img)
        
         # Prepare a linear color map
        cm = pg.colormap.get('CET-R4')
        
        # Insert a Colorbar, non-interactive with a label
        self.bar = pg.ColorBarItem(values = (0, self.signal_lim), colorMap=cm, interactive = False, label = 'RMS (\u03BCVolt)', )
        self.bar.setImageItem(self.img, insert_in = self.RealTimePlotWidget.window)
        
        # Scale factor (number of interpolation points to cover a width of n columns)
        corr_x = len(self._x_interpolate) / max(self._x_interpolate)
        
        # Scale factor (number of interpolation points to cover a width of n rows)
        corr_y = len(self._y_interpolate) / max(self._y_interpolate)
        
        # Number of rows/columns depends on grid-type
        locs = np.array([[(i%self.n_y) * corr_y, int(i/self.n_y) * corr_x] for i in range(self.n_x*self.n_y)])
         
        # Text object required for the tail orientation
        tail_text = pg.TextItem('TAIL', (128, 128, 128), anchor=(0, 0))
        tail_text.setFont(QtGui.QFont("Times", 16, QtGui.QFont.ExtraBold))
        
        # Initialise the standard format for the different indicators
        if self.tail_orientation == 'Left' or self.tail_orientation == 'left': 
            self.spots = [{'pos': locs[i], 'size': 5, 'pen': 'k', 'brush': QtGui.QBrush(QtGui.QColor(128, 128, 128))} \
                              for i in range(self._EMG_chans+len(self.dummy_chan)) if self.channel_conversion_list[i] <= self._EMG_chans]
            tail_text.setPos(self.spots[0]['pos'][0] - 1 * corr_y, self.spots[0]['pos'][1] + 1.5 * corr_x)
            
        elif self.tail_orientation == 'Up' or self.tail_orientation == 'up':
            self.spots = [{'pos': [locs.T[1][-i-1], locs[::-1].T[0][-i-1]], 'size': 5, 'pen': 'k', 'brush': QtGui.QBrush(QtGui.QColor(128, 128, 128))} \
                              for i in range(self._EMG_chans+len(self.dummy_chan)) if self.channel_conversion_list[i] <= self._EMG_chans]
            tail_text.setPos(self.spots[0]['pos'][0] - 2 * corr_y, self.spots[0]['pos'][1] - 0.5 * corr_x)
            
        elif self.tail_orientation == 'Right' or self.tail_orientation == 'right':
            self.spots = [{'pos': locs[::-1][i], 'size': 5, 'pen': 'k', 'brush': QtGui.QBrush(QtGui.QColor(128, 128, 128))} \
                              for i in range(self._EMG_chans+len(self.dummy_chan)) if self.channel_conversion_list[i] <= self._EMG_chans]
            tail_text.setPos(self.spots[0]['pos'][0] + 0.5 * corr_y, self.spots[0]['pos'][1] - 1.5 * corr_x)
            
        elif self.tail_orientation == 'Down' or self.tail_orientation == 'down':
            self.spots = [{'pos': [locs[::-1].T[1][-i-1], locs.T[0][-i-1]], 'size': 5, 'pen': 'k', 'brush': QtGui.QBrush(QtGui.QColor(128, 128, 128))} \
                              for i in range(self._EMG_chans+len(self.dummy_chan)) if self.channel_conversion_list[i] <= self._EMG_chans]
            tail_text.setPos(self.spots[0]['pos'][0] + 1 *corr_y, self.spots[0]['pos'][1] + 0.5 * corr_x)
            
        # Add the text object to the plot
        self.RealTimePlotWidget.window.addItem(tail_text)
         
        i_corr = 0
        # Set the position for each indicator
        for i in range(len(self.device._config._channels)):
            if i > (self.n_x*self.n_y) - 1:
                break
            # Select channel from conversion list and correct for dummy channels
            i_ch = self.channel_conversion_list[i]
            if i_ch <= (self._EMG_chans):
                # Place the name of each channel below the respective indicator
                # Text colour is red for disabled channels
                text = f'{self.device._config._channels[i_ch].alt_name: ^10}'
                if i_ch in self.channel_insertion_list:
                    t_item = pg.TextItem(text, (255, 0, 0), anchor=(0, 0))
                else:
                    t_item = pg.TextItem(text, (128, 128, 128), anchor=(0, 0))
                t_item.setPos(self.spots[i-i_corr]['pos'][0] -.25, self.spots[i-i_corr]['pos'][1] + .1)
                self.RealTimePlotWidget.window.addItem(t_item) 
            else:
                i_corr = i_corr + 1
    
                
        # Add all indicators to the plot
        self.c = pg.ScatterPlotItem(self.spots)
        self.RealTimePlotWidget.window.addItem(self.c)

        self.RealTimePlotWidget.window.invertY(True)
            
        
    @QtCore.Slot(object)
    def update_plot(self, data):
        """ Method that updates the indicators according to the measured impedance values
        """
        self.img.setImage(data, autoRange=False, autoLevels=False)

    def _update_scale(self, type_flag):
        """Method that creates a different range in which the heatmap is presented"""
        
        data = self.img.image
        if type_flag == 'scale':
            self.signal_lim = np.nanmax(data)
        elif type_flag == 'range':
            self.signal_lim = int(self.gui_handle.set_range_box.currentText())
        
        self.bar.setLevels(low = 0, high = self.signal_lim)


    def setupThread(self):
        """ Method that initialises the sampling thread of the device
        """
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
    """ Class responsible for sampling the data from the device
    """
    # Initialise the ouptut object
    output = QtCore.Signal(object)
    def __init__(self, main_class):
        QtCore.QObject.__init__(self)
        
        # Access initialised values from the GUI class
        self.device = main_class.device
        self.sample_rate = main_class.sample_rate
        self._EMG_chans = main_class._EMG_chans
        self.grid_type = main_class.grid_type
        self.channel_conversion_list = main_class.channel_conversion_list
        self._preprocess_wifi = main_class._preprocess_wifi
        
        self.window_buffer = np.zeros((self._EMG_chans+len(main_class.dummy_chan), self.sample_rate * 5))
        self.window_rms_size = self.sample_rate // 4
        self._add_final = 0
        
        self.sos = signal.butter(2, 10, 'highpass', fs=self.sample_rate, output='sos')
        z_sos0 = signal.sosfilt_zi(self.sos)
        self.z_sos = np.repeat(z_sos0[:, np.newaxis, :], (self._EMG_chans+len(main_class.dummy_chan)), axis=1)
    
        self.n_x=main_class.n_x
        self.n_y=main_class.n_y
        self._x_grid = np.arange(0, self.n_x, dtype=int)
        self._y_grid = np.arange(0, self.n_y, dtype=int)
                
        self._x_interpolate = main_class._x_interpolate
        self._y_interpolate = main_class._y_interpolate
        
        self.tail_orientation = main_class.tail_orientation
        self.channel_insertion_list = main_class.channel_insertion_list
        self.dummy_chan = main_class.dummy_chan
        
        # Prepare Queue
        self.q_sample_sets = queue.Queue(1000)
        
        # Register the consumer to the sample server
        sample_data_server.registerConsumer(self.device.id, self.q_sample_sets)
        
        # Start measurement
        self.sampling = True
        
    @QtCore.Slot()
    def update_samples(self): 
        """ Method that retrieves the sample data from the device. The method 
            gives the impedance value as output
        """
        
        while self.sampling:
            while not self.q_sample_sets.empty():
                
                # Retrieve sample data from the sample_data_server queue
                sd = self.q_sample_sets.get()
                self.q_sample_sets.task_done()
                
                # Reshape the samples retrieved from the queue
                samples = np.reshape(sd.samples, (sd.num_samples_per_sample_set, sd.num_sample_sets), order = 'F')
                self.new_samples = sd.num_sample_sets
                
                conversion_list = self.channel_conversion_list  
                # Missing samples are registered as NaN. This crashes the filter. 
                # Therefore, copies are inserted for the filtered data
                if self._preprocess_wifi:
                    find_nan = np.isnan(samples)
                    if find_nan.any():
                        idx_nan = np.where(np.isnan(samples))
                        samples[idx_nan] = samples[idx_nan[0], idx_nan[1][0]-1]

                # Insert disabled channels to maintain proper channel positions
                if len(self.channel_insertion_list) == 1:
                    samples = np.insert(samples, self.channel_insertion_list, 0, axis=0)
                else:
                    for ch in self.channel_insertion_list:
                        samples = np.insert(samples, ch, 0, axis=0)
                
                # Insert dummy channels
                for ch in self.dummy_chan:
                    samples = np.insert(samples, self._EMG_chans+1, np.nan, axis=0)
             
                # Fill the window buffer with the reshaped sample set
                self.window_buffer[:, self._add_final:(self._add_final + self.new_samples)] = samples[conversion_list,:]
                
                if self._add_final + self.new_samples > self.window_rms_size:
                    filt_data, self.z_sos = signal.sosfilt(self.sos, self.window_buffer[:, 0:self.window_rms_size], zi = self.z_sos)
                    
                    if self._preprocess_wifi:
                        if np.isnan(filt_data).any():
                            print('The filter crashed due to lost samples; resetting filter...')
                            self.sos = signal.butter(2, 10, 'highpass', fs=self.sample_rate, output='sos')
                            z_sos0 = signal.sosfilt_zi(self.sos)
                            self.z_sos = np.repeat(z_sos0[:, np.newaxis, :], self._EMG_chans, axis=1)
                         
                    rms_data = np.sqrt(np.mean(filt_data**2, axis = 1))
                   
                    # Reshape to 2d
                    rms_data = np.reshape(rms_data, (self.n_x,self.n_y)).T
                  
                    # Interpolation functions and mapping of dummy channels
                    # Dummy channels are NaN, apply dubble filter to handle this
                    nan_map = np.zeros_like(rms_data)
                    nan_map[ np.isnan(rms_data) ] = 1
                    
                    rms_filled = rms_data.copy()
                    rms_filled[ np.isnan(rms_data) ] = 0
                                         
                    f = interpolate.interp2d(self._x_grid, self._y_grid, rms_filled, kind='linear')
                    f_nan = interpolate.interp2d(self._x_grid, self._y_grid, nan_map, kind='linear')
                    
                    heatmap = f(self._x_interpolate, self._y_interpolate)
                    nan_heatmap = f_nan( self._x_interpolate, self._y_interpolate )
                    heatmap[ nan_heatmap>0 ] = np.nan
                   
                    # Rotate heatmap based on tail orientation
                    if self.tail_orientation == 'Left' or self.tail_orientation == 'left': 
                        output_heatmap = heatmap  
                    elif self.tail_orientation == 'Up' or self.tail_orientation == 'up':                      
                        output_heatmap = np.rot90(heatmap, 1)
                    elif self.tail_orientation == 'Right' or self.tail_orientation == 'right': 
                        output_heatmap = np.rot90(heatmap, 2)
                    elif self.tail_orientation == 'Down' or self.tail_orientation == 'down': 
                        output_heatmap = np.rot90(heatmap, 3)
                    
                    self._add_final = 0
                    
                    self.window_buffer = np.hstack((self.window_buffer[:,self.window_rms_size:], np.zeros(((self._EMG_chans+len(self.dummy_chan)), self.window_rms_size)) ))
                    self.output.emit(output_heatmap)
                    
                else:
                    self._add_final += self.new_samples
                
            # Pause the thread so that the update does not happen too fast
            time.sleep(0.01)
           
    def stop(self):
        """ Method that is executed when the thread is terminated. 
            This stop event stops the measurement and closes the connection to 
            the device.
        """
        # self.device.stop_measurement()
        self.sampling = False
        


if __name__ == "__main__":
    # Initialise the TMSi-SDK first before starting using it
    tmsi_device.initialize()
    
    # Create the device object to interface with the SAGA-system.
    dev = tmsi_device.create(tmsi_device.DeviceType.saga, DeviceInterfaceType.docked, DeviceInterfaceType.usb)

    # Find and open a connection to the SAGA-system and print its serial number
    dev.open()
    print("handle 1 " + str(dev.info.ds_serial_number))
    
    # Initialise the application
    app = QtWidgets.QApplication(sys.argv)
    # Define the GUI object and show it
    window = HDEMGPlot(figurename = 'An HDEMG Heatmap Plot', device = dev, tail_orientation='up')
    window.show()
    
    # Enter the event loop
    # sys.exit(app.exec_()) 
    app.exec_()
    dev.close()
