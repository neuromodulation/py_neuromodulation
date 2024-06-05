from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Signal, QThread
from py_neuromodulation import (nm_settings, nm_mnelsl_stream, nm_define_nmchannels, nm_stream_offline)
from mne_lsl.lsl import resolve_streams
import pandas as pd
from mne_lsl.stream import StreamLSL
import threading
import asyncio

possible_streams = resolve_streams()
possible_streams

exg_stream = possible_streams[0]
print(f'channel names: {exg_stream.get_channel_names()}')
print(exg_stream.get_channel_info)


settings = nm_settings.get_default_settings()
settings["features"]["welch"] = False
settings["features"]["fft"] = True
settings["features"]["bursts"] = False
settings["features"]["sharpwave_analysis"] = False
settings["features"]["coherence"] = False


# create an array ch_names with the channel names "ch0", "ch1", "ch2", ...
ch_names = []
ch_types = []
for i in range(exg_stream.n_channels):
    ch_names.append(f'ch{i}')
    ch_types.append(exg_stream.stype)

nm_channels = nm_define_nmchannels.set_channels(
    ch_names = ch_names,
    ch_types= ch_types,
    reference = "default",
    new_names = "default",
    used_types= "eeg"
)

stream_name = exg_stream.name
#  stream.run(stream_lsl=True, stream_lsl_name=stream_name)

classes = ['relax', 'clench']


class StreamWorker(QThread):
    update_signal = Signal(object)

    def __init__(self):
        super().__init__()


    def run(self):
        print("Running")
        print(stream_name)
        stream = nm_stream_offline.Stream(sfreq=exg_stream.sfreq, nm_channels=nm_channels, settings=settings, verbose = True, line_noise = 50)
        # stream = nm_mnelsl_stream.LSLStream(stream_name=stream_name, settings=settings)
        data = stream.run(stream_lsl_name = stream_name, stream_lsl=True)
        print(data)
        print("Data received")
        for i in range(1, 10000):
            # data = stream.get_next_batch()
            self.update_signal.emit(i)
            print(data)
            QThread.msleep(0.005) 


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.label = QLabel("Press the button to start the training")
        self.setWindowTitle("PyNeuromodulation Trainer")
        self.button = QPushButton("Start Training")
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

        self.button.clicked.connect(self.start_thread)

    def start_thread(self):
        self.thread = StreamWorker()
        self.thread.update_signal.connect(self.update_label)
        self.thread.start()

    def update_label(self, value):
        self.label.setText(f"Count: {value[0]}")

app = QApplication([])
window = MainWindow()
window.show()
app.exec()