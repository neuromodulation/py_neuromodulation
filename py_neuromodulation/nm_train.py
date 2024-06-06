from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Signal, QThread
from sklearn.metrics import classification_report, accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import make_pipeline
from py_neuromodulation import (nm_mnelsl_stream)
from py_neuromodulation import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class StreamWorker(QThread):
    update_signal = Signal(object)
    data_signal = Signal(pd.DataFrame)

    def __init__(self, stream, stream_name, classes):
        super().__init__()
        self.stream = stream
        self.stream_name = stream_name
        self.classes = classes

    def run(self):
        logger.info("Retrieving data from stream...")
        data = self.stream.run(stream_lsl_name = self.stream_name, stream_lsl=True)
        print(len(self.classes))
        data['task'] = len(self.classes)
        self.data_signal.emit(data)
        logger.info(data)
        logger.info("Data received")

class Trainer(QWidget):
    def __init__(self, model, stream, stream_name: str, classes: list | None = ['relax', 'perform task']):
        self.app = QApplication([])
        super().__init__()
        self.init_ui()
        self.stream = stream
        self.stream_name = stream_name
        self.classes = classes
        self.model = model
        self.df_train = pd.DataFrame()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.label = QLabel("Press the button to start the training")
        self.setWindowTitle("PyNeuromodulation Trainer")
        self.button = QPushButton("Start Recording")
        self.stop_button = QPushButton("Stop Recording")
        self.train_button = QPushButton("Train Model")
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.stop_button)
        self.layout.addWidget(self.train_button)
        self.setLayout(self.layout)
        self.button.clicked.connect(self.start_thread)
        self.stop_button.clicked.connect(self.stop_thread)
        self.train_button.clicked.connect(self.train_model)

        self.stop_button.setEnabled(False)
        self.train_button.setEnabled(False)

    def start_thread(self):
        self.label.setText(f"Recording started. Please {self.classes[0]}" )
        self.classes.pop(0)
        self.label.setStyleSheet("color: green;")
        self.thread = StreamWorker(self.stream, self.stream_name, self.classes)
        self.thread.update_signal.connect(self.update_label)
        self.thread.data_signal.connect(self.update_df)
        self.thread.start()
        self.stop_button.setEnabled(True)
        self.button.setEnabled(False)

    def update_df(self, data):
        logger.info ('Adding new data to the train dataframe')
        self.df_train = pd.concat([self.df_train, data], ignore_index=True)
        

    def stop_thread(self):
        self.label.setText("Recording stopped.")
        self.label.setStyleSheet("color: white;")
        self.stop_button.setEnabled(False)
        self.button.setEnabled(True)
        if len(self.classes) == 0:
            self.button.setEnabled(False)
            self.train_button.setEnabled(True)
        else:
            self.button.setEnabled(True)
        if hasattr(self, 'thread') and self.thread.isRunning(): 
            self.thread.stream.lsl_stream.disconnect()

    def update_label(self, value):
        self.label.setText(f"Count: {value}")


    def train_model(self):
        logger.info("Training model...")
        logger.info(f'Shape of training data: {self.df_train.shape}')

        max_value = self.df_train['task'].max()
        self.df_train['task'] = self.df_train['task'].apply(lambda x: max_value - x)

        df_filtered = self.df_train.drop(columns=['task', 'time'])
        features = df_filtered.columns[~df_filtered.columns.str.endswith('LineLength')]

        X = self.df_train[features]
        y = self.df_train['task']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        pipeline = make_pipeline(StandardScaler(), self.model)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    def start(self):
        self.show()
        self.app.exec()