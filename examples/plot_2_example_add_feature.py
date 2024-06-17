"""
===================
Adding New Features
===================

"""
# %%
import py_neuromodulation as nm
import numpy as np
from typing import Iterable

# %%
# In this example we will demonstrate how a new feature can be added to the existing feature pipeline.
# This can be done by creating a new feature class that implements the protocol class :class:`~nm_features.NMFeature`
# and registering it with the :func:`~nm_features.AddCustomFeature` function.


# %%
# Let's create a new feature class called `ChannelMean` that calculates the mean signal for each channel.
# We can optinally make it inherit from :class:`~nm_features.NMFeature` but as long as it has an adequate constructor
# and a `calc_feature` method with the appropriate signatures it will work.
# The :func:`__init__` method should take the settings, channel names and sampling frequency as arguments.
# The `calc_feature` method should take the data and a dictionary of features as arguments and return the updated dictionary.
class ChannelMean:
    def __init__(
        self, settings: nm.NMSettings, ch_names: Iterable[str], sfreq: float
    ) -> None:
        # If required for feature calculation, store the settings,
        # channel names and sampling frequency (optional)
        self.settings = settings
        self.ch_names = ch_names
        self.sfreq = sfreq

        # Here you can add any additional initialization code
        # For example, you could store parameters for the functions\
        # used in the calc_feature method
        
        self.feature_name = "channel_mean"

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        # Here you can add any feature calculation code
        # This example simply calculates the mean signal for each channel
        ch_means = np.mean(data, axis=1)

        # Store the calculated features in the features_compute dictionary
        # Be careful to use a unique keyfor each channel and metric you compute
        for ch_idx, ch in enumerate(self.ch_names):
            features_compute[f"{self.feature_name}_{ch}"] = ch_means[ch_idx]

        # Return the updated features_compute dictionary to the stream
        return features_compute


nm.AddCustomFeature("channel_mean", ChannelMean)

# %%
# Now we can instantiate settings and observe that the new feature has been added to the list of features
settings = nm.NMSettings() # Get default settings

settings.features

# %% 
# Let's create some artificial data to demonstrate the feature calculation.
N_CHANNELS = 5
N_SAMPLES = 10000 # 10 seconds of random data at 1000 Hz sampling frequency

data = np.random.random([N_CHANNELS, N_SAMPLES]) 
stream = nm.Stream(
    sfreq=1000,
    data=data,
    settings = settings,
    sampling_rate_features_hz=10,
    verbose=False,
)

feature_df = stream.run()
columns = [col for col in feature_df.columns if "channel_mean" in col]

feature_df[columns]


# %% 
# The new feature is added to the settings object 



