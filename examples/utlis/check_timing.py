import py_neuromodulation as pn
import numpy as np
import timeit

def get_fast_compute_settings():
    settings = pn.nm_settings.get_default_settings()
    #settings = pn.nm_settings.reset_settings(settings)
    #settings = pn.nm_settings.set_settings_fast_compute(settings)
    settings["preprocessing"] = ["re_referencing", "notch_filter"]
    #settings["features"]["fft"] = True
    settings["postprocessing"]["feature_normalization"] = False
    return settings


time_length = 100000
sampling_rate = 1000
ch_names = 20

data = np.random.random([ch_names, time_length])
stream = pn.Stream(sfreq=sampling_rate, data=data,
                   sampling_rate_features_hz=10,
                   verbose=True,
                   settings=get_fast_compute_settings()
)

f = stream.run()


