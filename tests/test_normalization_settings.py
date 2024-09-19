import pytest
import numpy as np

from py_neuromodulation import processing, NMSettings


NORM_METHODS = processing.NormalizationSettings.list_normalization_methods()


def test_raw_normalization_init():
    """Test to check that raw normalizer can't be instantiated with a wrong normalization method."""

    settings = NMSettings.get_default()
    settings.raw_normalization_settings.normalization_method = "meann"  # type: ignore

    with pytest.raises(Exception):
        processing.RawNormalizer(sfreq=1000, settings=settings)


def test_feature_normalization_init():
    """Test to check that feature normalizer can't be instantiated with a wrong normalization method."""

    settings = NMSettings.get_default()
    settings.feature_normalization_settings.normalization_method = "meann"  # type: ignore

    with pytest.raises(Exception):
        processing.FeatureNormalizer(settings=settings)


def test_process_norm_features():
    """Test that FeatureNormalizer returns correct values (not nan or infinite).
    Also test that the previous data is stored correctly."""

    settings = NMSettings.get_default()
    settings.feature_normalization_settings.normalization_method = "mean"

    norm = processing.FeatureNormalizer(settings=settings)

    data = np.ones([1, 5])
    data_normed = norm.process(data)

    assert np.all(np.isfinite(data_normed))
    assert np.allclose(data, norm.previous)


def test_previous_size_FeatureNorm():
    """Test that previous batch data is clipped correctly when clip is enabled (default clip = 3)"""
    norm = processing.FeatureNormalizer(settings=NMSettings())

    num_features = 5

    for _ in range(150):
        np.random.seed(0)
        norm.process(np.random.random([1, num_features]))

    assert norm.previous.shape[0] < norm.num_samples_normalize


def test_zscore_feature_analysis():
    """Test that previous data is not clipped  when clip is set to False"""

    settings = NMSettings.get_default()
    settings.feature_normalization_settings.clip = False

    norm = processing.FeatureNormalizer(settings=settings)

    num_features = 5

    data_to_norm = np.zeros([1, num_features])
    data_normed = np.zeros([1, num_features])

    for _ in range(400):
        np.random.seed(0)
        data_to_norm = np.random.random([1, num_features])
        data_normed = norm.process(data_to_norm)

    expect_res = (
        norm.previous[:, 0].std() * data_normed[0, 0] + norm.previous[:, 0].mean()
    )

    assert pytest.approx(expect_res, 0.1) == data_to_norm[0, 0]


def test_zscore_raw_analysis():
    """Test that zscore is giving the expected results"""

    settings = NMSettings.get_default()
    settings.raw_normalization_settings.clip = False

    norm = processing.RawNormalizer(sfreq=10, settings=settings)

    num_samples = 100
    data_to_norm = np.zeros([1, num_samples])
    data_normed = np.zeros([1, num_samples])

    for _ in range(400):
        data_to_norm = np.random.random([1, num_samples])
        data_normed = norm.process(data_to_norm)

    expect_res = (
        norm.previous[:, 0].std() * data_normed[0, 0] + norm.previous[:, 0].mean()
    )

    assert np.allclose(expect_res, data_to_norm[0, 0], rtol=0.1, atol=0.1)


def test_all_norm_methods_raw():
    """Test that all raw normalization methods return correct values (not nan or infinite)"""

    settings = NMSettings.get_default()
    settings.raw_normalization_settings.clip = False

    for norm_method in NMSettings.list_normalization_methods():
        settings.raw_normalization_settings.normalization_method = norm_method

        norm = processing.RawNormalizer(sfreq=10, settings=settings)

        num_samples = 10

        data_to_norm = np.zeros([1, num_samples])
        data_normed = np.zeros([1, num_samples])

        for _ in range(10):
            np.random.seed(0)
            data_to_norm = np.random.random([1, num_samples])
            data_normed = norm.process(data_to_norm)

        assert np.all(np.isfinite(data_normed))


def test_all_norm_methods_feature():
    """Test that all feature normalization methods return correct values (not nan or infinite)"""

    settings = NMSettings.get_default()
    settings.feature_normalization_settings.clip = False

    for norm_method in NMSettings.list_normalization_methods():
        settings.feature_normalization_settings.normalization_method = norm_method

        norm = processing.FeatureNormalizer(settings=settings)

        num_samples = 10

        data_to_norm = np.zeros([1, num_samples])
        data_normed = np.zeros([1, num_samples])

        for i in range(10):
            np.random.seed(i)
            data_to_norm = np.random.random([1, num_samples])
            data_normed = norm.process(data_to_norm)

        assert np.all(np.isfinite(data_normed))
