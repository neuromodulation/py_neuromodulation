import multiprocessing
from pyneuromodulation import nm_RealTimeClientStream
from pynput.keyboard import Key, Listener
import time
from mne_realtime import LSLClient
import numpy as np
import pandas as pd
import sys
import os
import signal

# Check HOST_Name by calling in OpenBCI_LSL:
# python openbci_lsl.py --stream 
HOST_NAME = "openbci_eeg_id109"

def getData(queue_raw):

    def on_press(key):
        print('{0} pressed'.format(
            key))

    def on_release(key):
        if key == Key.esc:
            # Stop listener
            print("reiceived stop key pressed")
            queue_raw.put(None)
            return False

    listener = Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    with LSLClient(host=HOST_NAME, wait_max=10) as client:
        print("client initialized")
        client_info = client.get_measurement_info()
        sfreq = int(client_info['sfreq'])

        while listener.is_alive() is True:
            ieeg_batch = np.squeeze(client.get_data_as_epoch(n_samples=sfreq))
            queue_raw.put(ieeg_batch)

def calcFeatures(queue_raw, queue_features):
    rt_estimator = nm_RealTimeClientStream.RealTimePyNeuro()
    FLAG_STOP = False
    while FLAG_STOP is False:
        ieeg_batch = queue_raw.get()  # ch, samples
        if ieeg_batch is None:
            queue_features.put(None)
            FLAG_STOP = True
        else:
            features = rt_estimator.call_run(ieeg_batch)
            print("calc features")
            queue_features.put(features)

def sendFeatures(queue_features):
    
    features_out = pd.DataFrame()
    FLAG_STOP = False
    while FLAG_STOP is False:
        features = queue_features.get()
        if features is None:
            features_out.to_csv("reatime_features.csv")
            print("SAVED")
            FLAG_STOP = True
        else:
            features_out = features_out.append(features, ignore_index=True)
            print("length of features:" + str(len(features_out)))

if __name__ == "__main__":

    # Make queue only get a single item
    # explained here: https://stackoverflow.com/questions/69992497/how-to-detect-a-pressed-key-within-python-process
    queue_raw = multiprocessing.Queue(1)
    queue_features = multiprocessing.Queue()

    processes = [
        multiprocessing.Process(target=getData, args=(queue_raw,)),
        multiprocessing.Process(target=calcFeatures, args=(queue_raw, queue_features,)),
        multiprocessing.Process(target=sendFeatures, args=(queue_features,))
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
    