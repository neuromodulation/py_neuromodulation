"""
Real-time feature estimation
============================

"""

# %%
# Implementation of individual nm_streams
# ---------------------------------------
#
# *py_neuromodulation* was optimized for computation of real-time data streams.
# There are however center -and lab specific hardware acquisition systems. Therefore, each experiment requires modules to interact with hardware platforms
# which periodically acquire data.
#
# Given the raw data, data can be analyzed using *py_neuromodulation*. Preprocessing methods, such as re-referencing and normalization,
# feature computation and decoding can be performed then in real-time.
#
# For online as well as as offline analysis, the :class:`~nm_stream_abc` class needs to be instantiated.
# Here the `nm_settings` and `nm_channels` are required to be defined.
# Previously for the offline analysis, an offline :class:`~nm_generator` object was defined that periodically yielded data.
# For online data, the :meth:`~nm_stream_abc.run` function therefore needs to be overwritten, which first acquires data and then calls
# the :meth:`~nm_run_analysis.process` function.
#
# The following illustrates in pseudo-code how such a stream could be initialized:
#
# .. code-block:: python
#
#    from py_neuromodulation import nm_stream_abc
#
#    class MyStream(nm_stream_abc):
#    def __init__(self, settings, channels):
#        super().__init__(settings, channels)
#
#    def run(self):
#       features_ = []
#        while True:
#           data = self.acquire_data()
#           features_.append(self.run_analysis.process(data))
#           # potentially use machine learning model for decoding
#
#
# Computation time examples
# -------------------------
#
# The following example calculates for six channels, CAR re-referencing, z-score normalization and FFT features results the following computation time:

# %%
import py_neuromodulation as nm
from py_neuromodulation import nm_settings
import numpy as np
import timeit


def get_fast_compute_settings():
    settings = nm_settings.get_default_settings()
    settings = nm_settings.reset_settings(settings)
    settings = nm_settings.set_settings_fast_compute(settings)
    settings["preprocessing"] = ["re_referencing", "notch_filter"]
    settings["features"]["fft"] = True
    settings["postprocessing"]["feature_normalization"] = True
    return settings


data = np.random.random([1, 1000])

print("FFT Features, CAR re-referencing, z-score normalization")
print()
print("Computation time for single ECoG channel: ")
stream = nm.Stream(
    sfreq=1000,
    data=data,
    sampling_rate_features_hz=10,
    verbose=False,
    settings=get_fast_compute_settings(),
)
print(
    f"{np.round(timeit.timeit(lambda: stream.data_processor.process(data), number=100)/100, 3)} s"
)

print("Computation time for 6 ECoG channels: ")
data = np.random.random([6, 1000])
stream = nm.Stream(
    sfreq=500,
    data=data,
    sampling_rate_features_hz=10,
    verbose=False,
    settings=get_fast_compute_settings(),
)
print(
    f"{np.round(timeit.timeit(lambda: stream.data_processor.process(data), number=100)/100, 3)} s"
)

print(
    "\nFFT Features & Temporal Waveform Shape & Hjorth & Bursts, CAR re-referencing, z-score normalization"
)
print("Computation time for single ECoG channel: ")
data = np.random.random([1, 1000])
stream = nm.Stream(
    sfreq=1000, data=data, sampling_rate_features_hz=10, verbose=False
)
print(
    f"{np.round(timeit.timeit(lambda: stream.data_processor.process(data), number=10)/10, 3)} s"
)


# %%
# Those results show that the computation time for a typical pipeline (FFT, re-referencing, notch-filtering, feature normalization)
# is well below 10 ms, which is fast enough for real-time analysis with feature sampling rates below 100 Hz.
# Computation of more complex features could still result in feature sampling rates of more than 30 Hz.
#
# Real-time movement decoding using the TMSi-SAGA amplifier
# ---------------------------------------------------------
#
# In the following example, we will show how we setup a real-time movement decoding experiment using the TMSi-SAGA amplifier.
# First, we relied on different software modules for data streaming and visualization.
# `LabStreamingLayer <https://labstreaminglayer.org>`_ allows for real-time data streaming and synchronization across multiple devices.
# We used `timeflux <https://timeflux.io>`_ for real-time data visualization of features, decoded output.
# For raw data visualization we used `Brain Streaming Layer <https://fcbg-hnp-meeg.github.io/bsl/dev/index.html>`_.
#
# The code for real-time movement decoding is added in the GitHub branch `realtime_decoding <https://github.com/neuromodulation/py_neuromodulation/tree/realtime_decoding>`_.
# Here we relied on the `TMSI SAGA Python interface <https://gitlab.com/tmsi/tmsi-python-interface>`_.
#
