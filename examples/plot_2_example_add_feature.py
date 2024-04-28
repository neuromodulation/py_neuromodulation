"""
===================
Adding New Features 
===================

"""

import py_neuromodulation as nm
from py_neuromodulation import nm_features_abc
import numpy as np
from typing import Iterable

# %%
# In this example we will demonstrate how a new feature can be added to the existing feature pipeline.
# This can be done simply by adding an object of the inherited :class:`~nm_features_abc.Feature`
# class to the stream `stream.run_analysis.features.features` list.

data = np.random.random([1, 1000])

stream = nm.Stream(
    sfreq=1000,
    data=data,
    sampling_rate_features_hz=10,
    verbose=False,
)


class NewFeature(nm_features_abc.Feature):
    def __init__(
        self, settings: dict, ch_names: Iterable[str], sfreq: float
    ) -> None:
        self.s = settings
        self.ch_names = ch_names

    def calc_feature(self, data: np.ndarray, features_compute: dict) -> dict:
        for ch_idx, ch in enumerate(self.ch_names):
            features_compute[f"new_feature_{ch}"] = np.mean(data[ch_idx, :])

        return features_compute

    def test_settings():
        pass


newFeature = NewFeature(
    stream.settings, list(stream.nm_channels["name"]), stream.sfreq
)
stream.run_analysis.features.features.append(newFeature)

features = stream.run_analysis.process(data)
feature_name = f"new_feature_{stream.nm_channels['name'][0]}"

print(f"{feature_name}: {features[feature_name]}")

# %%
# This example shows a simple newly instantiated feature class called `NewFeature`.
# The instantiated `newFeature` object could then be added to the existing feature list by calling
# `stream.run_analysis.features.features.append(newFeature)`.
#
# To permanently add a novel feature, the new feature class needs to be added to
# the :class:`~nm_features` class. This can be done by inserting the feature_name in
# in the :class:`~nm_features.Feature` init function:
#
# .. code-block:: python
#
#    for feature in s["features"]:
#        if s["features"][feature] is False:
#            continue
#        match feature:
#            case "new_feature":
#                FeatureClass = nm_new_feature.NewFeature
#            ...
#
# The new feature class can then be used by setting the `settings["feature"]["new_feature"]` value in the
# settings to true.
#
