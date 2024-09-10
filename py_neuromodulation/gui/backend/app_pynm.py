import py_neuromodulation as nm
import numpy as np


class PyNMState:
    def __init__(self) -> None:
        self.stream: nm.Stream = nm.Stream(sfreq=1500, data=np.random.random([15, 15]))
        self.settings: nm.NMSettings = nm.NMSettings(sampling_rate_features=17)
