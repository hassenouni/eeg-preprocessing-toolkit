import numpy as np
import pytest
from eeg_preprocessing.pipeline import (
    bandpass,
    notch,
    resample,
    average_reference,
    quality_flags,
    ChannelStandardizer,
)


def test_bandpass_frequency_selectivity():
    fs = 256
    t = np.arange(fs * 12) / fs
    x = (np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 80 * t))[None, :]
    y = bandpass(x, fs, 5, 30)[0, fs:-fs]
    tt = t[fs:-fs]

    def amplitude(f):
        return abs(np.mean(y * np.exp(-2j * np.pi * f * tt))) * 2

    assert amplitude(10) > 0.9
    assert amplitude(80) < 0.01


def test_notch_rejects_line_component():
    fs = 256
    t = np.arange(fs * 10) / fs
    x = np.sin(2 * np.pi * 50 * t)[None, :]
    y = notch(x, fs)[:, fs:-fs]
    assert y.std() < 0.03


def test_resampling_suppresses_alias():
    fs = 256
    t = np.arange(fs * 8) / fs
    x = np.array([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 100 * t)])
    y = resample(x, fs, 64)
    assert y.shape == (2, 512)
    assert y[0, 64:-64].std() > 0.6
    assert y[1, 64:-64].std() < 0.02


def test_reference_and_quality():
    x = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]]) * 1e-6
    np.testing.assert_allclose(average_reference(x).mean(axis=0), 0, atol=1e-20)
    assert quality_flags(x)["flat"].tolist() == [True, False]


def test_normalization_never_refits_on_test_data():
    train = np.array([[0.0, 2.0, 4.0], [3.0, 3.0, 3.0]])
    scaler = ChannelStandardizer().fit(train)
    mean = scaler.mean_.copy()
    transformed = scaler.transform(train + 100)
    np.testing.assert_array_equal(scaler.mean_, mean)
    assert transformed.mean() > 10
    with pytest.raises(RuntimeError):
        ChannelStandardizer().transform(train)


def test_invalid_cutoffs():
    with pytest.raises(ValueError):
        bandpass(np.ones((2, 100)), 100, 1, 60)
