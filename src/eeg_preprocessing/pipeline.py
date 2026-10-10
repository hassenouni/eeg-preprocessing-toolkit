"""Offline filtering and explicit train-only normalization."""

from __future__ import annotations
from fractions import Fraction
import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt, iirnotch, filtfilt, lfilter, resample_poly
from eeg_dataset_explorer.signals import signal_array, sampling_rate


def bandpass(
    data: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0, order: int = 4, causal: bool = False
) -> np.ndarray:
    """Butterworth SOS filter along time. Default is offline zero-phase."""
    x = signal_array(data)
    sampling_rate(fs)
    if not (0 < low < high < fs / 2) or not isinstance(order, int) or order < 1:
        raise ValueError("Require 0 < low < high < Nyquist and positive integer order")
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfilt(sos, x, axis=-1) if causal else sosfiltfilt(sos, x, axis=-1)


def notch(
    data: np.ndarray, fs: float, frequency: float = 50.0, quality: float = 30.0, causal: bool = False
) -> np.ndarray:
    """Remove one line-frequency component; quality controls notch bandwidth."""
    x = signal_array(data)
    sampling_rate(fs)
    if not 0 < frequency < fs / 2 or not np.isfinite(quality) or quality <= 0:
        raise ValueError("Invalid notch frequency or quality")
    b, a = iirnotch(frequency, quality, fs)
    return lfilter(b, a, x, axis=-1) if causal else filtfilt(b, a, x, axis=-1)


def resample(data: np.ndarray, fs: float, target_fs: float) -> np.ndarray:
    """Anti-aliased polyphase resampling, rejecting poorly approximated ratios."""
    x = signal_array(data)
    sampling_rate(fs)
    sampling_rate(target_fs)
    ratio = Fraction(target_fs / fs).limit_denominator(10000)
    if not np.isclose(float(ratio), target_fs / fs, rtol=1e-8):
        raise ValueError("Sampling-rate ratio cannot be represented accurately")
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=-1)


def average_reference(data: np.ndarray) -> np.ndarray:
    """Subtract the instantaneous channel mean; select EEG channels beforehand."""
    x = signal_array(data)
    return x - x.mean(axis=0, keepdims=True)


def quality_flags(
    data: np.ndarray, flat_std: float = 1e-8, amplitude_limit: float = 500e-6
) -> dict[str, np.ndarray]:
    """Flag flat or large-amplitude channels; input and thresholds are volts."""
    x = signal_array(data)
    if not np.isfinite([flat_std, amplitude_limit]).all() or flat_std < 0 or amplitude_limit <= 0:
        raise ValueError("Invalid quality thresholds")
    return {"flat": x.std(axis=-1) <= flat_std, "large_amplitude": np.max(abs(x), axis=-1) > amplitude_limit}


class ChannelStandardizer:
    """Fit channel statistics on training samples only, then transform held-out data."""

    def fit(self, data: np.ndarray) -> ChannelStandardizer:
        x = signal_array(data)
        self.mean_ = x.mean(axis=-1, keepdims=True)
        self.scale_ = x.std(axis=-1, keepdims=True)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("Call fit on training data first")
        x = signal_array(data)
        if len(x) != len(self.mean_):
            raise ValueError("Channel count differs from fitted data")
        return (x - self.mean_) / self.scale_
