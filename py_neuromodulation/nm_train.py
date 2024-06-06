from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Signal, QThread
from py_neuromodulation import (nm_settings, nm_mnelsl_stream, nm_define_nmchannels, nm_stream_offline)
from mne_lsl.lsl import resolve_streams
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class StreamWorker(QThread):
    update_signal = Signal(object)

    def __init__(self, stream, stream_name):
        super().__init__()
        self.stream = stream
        self.stream_name = stream_name

    def run(self):
        print("Running")
        # stream = nm_mnelsl_stream.LSLStream(stream_name=stream_name, settings=settings)
        data = self.stream.run(stream_lsl_name = self.stream_name, stream_lsl=True)
        print(data)
        print("Data received")

class Trainer(QWidget):
    def __init__(self, stream, stream_name):
        super().__init__()
        self.init_ui()
        self.stream = stream
        self.stream_name = stream_name

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.label = QLabel("Press the button to start the training")
        self.setWindowTitle("PyNeuromodulation Trainer")
        self.button = QPushButton("Start Training")
        self.stop_button = QPushButton("Stop Training")
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.stop_button)
        self.setLayout(self.layout)
        self.button.clicked.connect(self.start_thread)
        self.stop_button.clicked.connect(self.stop_thread)

    def start_thread(self):
        self.label.setText("Training started")
        self.thread = StreamWorker(self.stream, self.stream_name)
        self.thread.update_signal.connect(self.update_label)
        self.thread.start()

    def stop_thread(self):
        self.label.setText("Training started")
        if hasattr(self, 'thread') and self.thread.isRunning(): 
            print("Stopping thread")
            self.thread.stream.lsl_stream.disconnect()
            # self.thread.terminate() 

    def update_label(self, value):
        self.label.setText(f"Count: {value}")


    # def train_model(self):

    #     df_comb = pd.concat([df_relax, df_clench], ignore_index=True)

    #     df_filtered = df_comb.drop(columns=['clench', 'time'])
    #     features = df_filtered.columns[~df_filtered.columns.str.endswith('LineLength')]

    #     X = df_comb[features]
    #     y = df_comb['clench']

    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def start(self):
        window = Trainer(self.stream, self.stream_name)
        window.show()
        app.exec()

app = QApplication([])