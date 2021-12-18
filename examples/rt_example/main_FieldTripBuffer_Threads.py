import threading
import queue
from pyneuromodulation import nm_RealTimeClientStream
from pynput.keyboard import Key, Listener
import time
from pyneuromodulation import FieldTrip
import numpy as np
import pandas as pd
import sys
import os
import signal

def getData(queue_raw, ftc):

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

    while listener.is_alive() is True:
        # read new data
        print('Trying to read last sample...')
        ieeg_batch = ftc.getData().T
        ieeg_batch = ieeg_batch[-128:,:]  # take last 
        queue_raw.put(ieeg_batch)
    ftc.disconnect()

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

def sendFeatures(queue_features, ftc):
    
    time_start = time.time()
    features_out = pd.DataFrame()
    FLAG_STOP = False
    while FLAG_STOP is False:
        features = queue_features.get()
        features["time"] = time.time() - time_start
        if features is None:
            features_out.to_csv("reatime_features.csv")
            print("SAVED")
            FLAG_STOP = True
        else:
            features_out = features_out.append(features, ignore_index=True)
            print("length of features:" + str(len(features_out)))
            
            # channel names are 1. data 2. ident 3. timestamp
            # put at the end of the buffer the calculated features of the data channel
            #H = ftc.getHeader()
            #to_send = np.zeros([H.nSamples, H.nChannels])
            #to_send[-features.shape[0]:,0] = np.array(features)
            #ftc.putData(to_send)

if __name__ == "__main__":

    # Make queue only get a single item
    # explained here: https://stackoverflow.com/questions/69992497/how-to-detect-a-pressed-key-within-python-process
    queue_raw = queue.Queue(1)
    queue_features = queue.Queue()
    
    # Python FieldTripBuffer https://www.fieldtriptoolbox.org/development/realtime/buffer_python/
    ftc = FieldTrip.Client()
    ftc.connect('localhost', 1972)    # might throw IOError
    H = ftc.getHeader()

    if H is None:
        print('Failed to retrieve header!')
        sys.exit(1)
    
    print(H)
    print(H.labels)

    threads = [
        threading.Thread(target=getData, args=(queue_raw,ftc,)),
        threading.Thread(target=calcFeatures, args=(queue_raw, queue_features,)),
        threading.Thread(target=sendFeatures, args=(queue_features,ftc,))
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    