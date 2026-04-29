"""
Signal Quality Index (SQI)
===========================
Composite signal quality assessment for ECG windows.
"""

import numpy as np
from scipy.signal import welch
from typing import Tuple, Optional


def kurtosis_sqi(signal: np.ndarray) -> float:
    """Kurtosis-based SQI. Clean QRS → high kurtosis. Score in [0,1]."""
    if len(signal) < 4:
        return 0.0
    std = np.std(signal)
    if std < 1e-10:
        return 0.0
    kurt = np.mean(((signal - np.mean(signal)) / std) ** 4)
    return float(np.clip((kurt - 3.0) / 12.0, 0.0, 1.0))


def spectral_sqi(signal: np.ndarray, fs: int) -> float:
    """QRS band (5-15Hz) energy ratio. Score in [0,1]."""
    if len(signal) < fs:
        return 0.5
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs * 2))
    total = np.sum(psd)
    if total < 1e-10:
        return 0.0
    qrs_power = np.sum(psd[(freqs >= 5) & (freqs <= 15)])
    return float(np.clip(qrs_power / total, 0.0, 1.0))


def power_sqi(signal: np.ndarray, fs: int) -> float:
    """ECG band (0.5-40Hz) power ratio. Score in [0,1]."""
    if len(signal) < fs:
        return 0.5
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs * 2))
    total = np.sum(psd)
    if total < 1e-10:
        return 0.0
    ecg_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 40)])
    return float(np.clip(ecg_power / total, 0.0, 1.0))


def beat_agreement_sqi(peaks_a: np.ndarray, peaks_b: np.ndarray, tolerance: int = 20) -> float:
    """F1 agreement between two R-peak detectors. Score in [0,1]."""
    if len(peaks_a) == 0 and len(peaks_b) == 0:
        return 1.0
    if len(peaks_a) == 0 or len(peaks_b) == 0:
        return 0.0
    matched_b = set()
    tp = 0
    for pa in peaks_a:
        diffs = np.abs(peaks_b - pa)
        j = np.argmin(diffs)
        if diffs[j] <= tolerance and j not in matched_b:
            tp += 1
            matched_b.add(j)
    prec = tp / len(peaks_a)
    rec = tp / len(peaks_b)
    if prec + rec < 1e-10:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def compute_sqi(
    signal: np.ndarray, fs: int,
    peaks_a: Optional[np.ndarray] = None,
    peaks_b: Optional[np.ndarray] = None,
    weights: Tuple[float, float, float, float] = (0.3, 0.3, 0.2, 0.2),
) -> Tuple[float, dict]:
    """Compute composite SQI from 4 components."""
    k = kurtosis_sqi(signal)
    s = spectral_sqi(signal, fs)
    p = power_sqi(signal, fs)
    b = beat_agreement_sqi(peaks_a, peaks_b) if peaks_a is not None and peaks_b is not None else 0.5
    composite = weights[0]*k + weights[1]*s + weights[2]*p + weights[3]*b
    return float(composite), {"kSQI": k, "sSQI": s, "pSQI": p, "bSQI": b, "composite": composite}
