import os
import unittest
import pytest
import numpy as np

from py_neuromodulation import nm_normalization


def test_raw_normalization_init():

    with pytest.raises(Exception):
        nm_normalization.RawNormalizer(
            sfreq=1000,
            sampling_rate_features_hz=500,
            normalization_method="meann",
            normalization_time_s=30,
            clip=3,
        )


def test_feature_normalization_init():

    with pytest.raises(Exception):
        nm_normalization.FeatureNormalizer(
            sampling_rate_features_hz=500,
            normalization_method="meann",
            normalization_time_s=30,
            clip=3,
        )


def test_process_norm_features():

    norm = nm_normalization.FeatureNormalizer(
        sampling_rate_features_hz=500,
        normalization_method="mean",
        normalization_time_s=30,
        clip=3,
    )
    data = np.ones([1, 5])
    data_normed = norm.process(data)

    assert np.all(np.isfinite(data_normed) == True)

    assert np.all(np.equal(data, norm.previous) == 1)


def test_previous_size_FeatureNorm():

    norm = nm_normalization.FeatureNormalizer(
        sampling_rate_features_hz=10,
        normalization_method="zscore",
        normalization_time_s=10,
        clip=3,
    )

    num_features = 5

    for _ in range(150):
        np.random.seed(0)
        data = norm.process(np.random.random([1, num_features]))

    assert norm.previous.shape[0] < norm.num_samples_normalize


def test_zscore_feature_analysis():
    norm = nm_normalization.FeatureNormalizer(
        sampling_rate_features_hz=10,
        normalization_method="zscore",
        normalization_time_s=30,
        clip=False,
    )

    num_features = 5

    for _ in range(400):
        np.random.seed(0)
        data_to_norm = np.random.random([1, num_features])
        data_normed = norm.process(data_to_norm)

    expect_res = (
        norm.previous[:, 0].std() * data_normed[0, 0]
        + norm.previous[:, 0].mean()
    )

    assert pytest.approx(expect_res, 0.1) == data_to_norm[0, 0]


def test_zscore_raw_analysis():

    norm = nm_normalization.RawNormalizer(
        sampling_rate_features_hz=10,
        normalization_method="zscore",
        normalization_time_s=30,
        sfreq=10,
        clip=False,
    )

    num_samples = 100

    for _ in range(400):
        data_to_norm = np.random.random([1, num_samples])
        data_normed = norm.process(data_to_norm)

    expect_res = (
        norm.previous[:, 0].std() * data_normed[0, 0]
        + norm.previous[:, 0].mean()
    )

    np.testing.assert_allclose(
        expect_res, data_to_norm[0, 0], rtol=0.1, atol=0.1
    )


def test_all_norm_methods_raw():

    for norm_method in [e.value for e in nm_normalization.NORM_METHODS]:
        norm = nm_normalization.RawNormalizer(
            sampling_rate_features_hz=10,
            normalization_method=norm_method,
            normalization_time_s=30,
            sfreq=10,
            clip=False,
        )

        num_samples = 10

        for _ in range(10):
            np.random.seed(0)
            data_to_norm = np.random.random([1, num_samples])
            data_normed = norm.process(data_to_norm)

        assert np.all(np.isfinite(data_normed) == True)


def test_all_norm_methods_feature():

    for norm_method in [e.value for e in nm_normalization.NORM_METHODS]:
        norm = nm_normalization.FeatureNormalizer(
            sampling_rate_features_hz=10,
            normalization_method=norm_method,
            normalization_time_s=30,
            clip=False,
        )

        num_samples = 10

        for i in range(10):
            np.random.seed(i)
            data_to_norm = np.random.random([1, num_samples])
            data_normed = norm.process(data_to_norm)

        assert np.all(np.isfinite(data_normed) == True)
