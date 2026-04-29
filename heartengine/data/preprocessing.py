"""
ECG Signal Preprocessing
=========================
Multi-method preprocessing pipeline: bandpass filtering, baseline wander
removal, powerline notch, and normalization.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
from scipy.ndimage import median_filter
from math import gcd
import pywt
from typing import Optional


def butterworth_bandpass(
    signal: np.ndarray,
    fs: int,
    low: float = 0.5,
    high: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter."""
    nyq = fs / 2
    low_norm = low / nyq
    high_norm = high / nyq

    # Clamp to valid range
    low_norm = max(low_norm, 1e-5)
    high_norm = min(high_norm, 0.9999)

    b, a = butter(order, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, signal, padlen=min(3 * max(len(b), len(a)), len(signal) - 1))


def notch_filter(
    signal: np.ndarray,
    fs: int,
    freq: float = 50.0,
    quality: float = 30.0,
) -> np.ndarray:
    """Remove powerline interference (50Hz or 60Hz)."""
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, signal, padlen=min(len(signal) - 1, 3 * max(len(b), len(a))))


def remove_baseline_wander_wavelet(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 9,
) -> np.ndarray:
    """
    Remove baseline wander using Stationary Wavelet Transform (SWT).
    Zeroes out the approximation coefficients at the deepest level,
    which captures the lowest frequency baseline drift.
    """
    # SWT requires signal length to be a multiple of 2^level
    orig_len = len(signal)
    pad_len = int(np.ceil(orig_len / (2 ** level))) * (2 ** level)
    padded = np.pad(signal, (0, pad_len - orig_len), mode="edge")

    # Decompose
    coeffs = pywt.swt(padded, wavelet, level=level)

    # Zero out approximation coefficients (lowest frequency = baseline)
    coeffs[-1] = (np.zeros_like(coeffs[-1][0]), coeffs[-1][1])

    # Reconstruct
    reconstructed = pywt.iswt(coeffs, wavelet)
    return reconstructed[:orig_len]


def remove_baseline_wander_median(
    signal: np.ndarray,
    fs: int,
    window_ms: int = 600,
) -> np.ndarray:
    """
    Remove baseline wander using cascaded median filters.
    Two-stage: 200ms (QRS width) then 600ms (T-wave width).
    """
    w1 = max(3, int(0.2 * fs) | 1)  # Ensure odd
    w2 = max(3, int(window_ms / 1000 * fs) | 1)

    baseline = median_filter(signal, size=w1)
    baseline = median_filter(baseline, size=w2)
    return signal - baseline


def normalize_signal(
    signal: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:
    """Normalize ECG signal."""
    if method == "zscore":
        mu = np.mean(signal)
        std = np.std(signal)
        if std < 1e-8:
            return signal - mu
        return (signal - mu) / std
    elif method == "minmax":
        mn, mx = np.min(signal), np.max(signal)
        rng = mx - mn
        if rng < 1e-8:
            return np.zeros_like(signal)
        return (signal - mn) / rng
    elif method == "robust":
        med = np.median(signal)
        iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
        if iqr < 1e-8:
            return signal - med
        return (signal - med) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_ecg(
    signal: np.ndarray,
    fs: int,
    bandpass: bool = True,
    remove_baseline: bool = True,
    remove_powerline: bool = True,
    normalize: bool = True,
    baseline_method: str = "wavelet",
    bandpass_low: float = 0.5,
    bandpass_high: float = 40.0,
    powerline_freq: float = 50.0,
    norm_method: str = "zscore",
) -> np.ndarray:
    """
    Full preprocessing pipeline for single-lead ECG.

    Pipeline order: Powerline removal → Bandpass → Baseline removal → Normalization
    """
    out = signal.copy().astype(np.float64)

    # 1. Powerline removal (before bandpass to avoid filter ringing)
    if remove_powerline:
        out = notch_filter(out, fs, freq=powerline_freq)

    # 2. Bandpass filter
    if bandpass:
        out = butterworth_bandpass(out, fs, low=bandpass_low, high=bandpass_high)

    # 3. Baseline wander removal
    if remove_baseline:
        if baseline_method == "wavelet":
            out = remove_baseline_wander_wavelet(out)
        elif baseline_method == "median":
            out = remove_baseline_wander_median(out, fs)

    # 4. Normalization
    if normalize:
        out = normalize_signal(out, method=norm_method)

    return out


def resample_signal(signal: np.ndarray, fs_orig: int, fs_target: int) -> np.ndarray:
    """Resample signal to target sampling rate."""
    if fs_orig == fs_target:
        return signal
    g = gcd(fs_target, fs_orig)
    return resample_poly(signal, fs_target // g, fs_orig // g)
