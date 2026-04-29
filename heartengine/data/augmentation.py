"""
ECG-Specific Data Augmentation
=================================
Realistic augmentations that preserve diagnostic features.
"""

import numpy as np
from typing import Optional, Tuple


def amplitude_scale(signal: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """Random amplitude scaling."""
    scale = np.random.uniform(*scale_range)
    return signal * scale


def temporal_shift(signal: np.ndarray, max_shift_samples: int = 50) -> np.ndarray:
    """Random temporal shift (circular)."""
    shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
    return np.roll(signal, shift)


def add_gaussian_noise(signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Add Gaussian noise at specified SNR."""
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def add_baseline_wander(signal: np.ndarray, fs: int, amplitude: float = 0.3) -> np.ndarray:
    """Add synthetic baseline wander (low-frequency sinusoid)."""
    t = np.arange(len(signal)) / fs
    freq = np.random.uniform(0.1, 0.5)
    phase = np.random.uniform(0, 2 * np.pi)
    wander = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return signal + wander


def add_emg_noise(signal: np.ndarray, amplitude: float = 0.05) -> np.ndarray:
    """Add muscle artifact (high-frequency noise bursts)."""
    noise = np.zeros_like(signal)
    n_bursts = np.random.randint(1, 4)
    for _ in range(n_bursts):
        burst_len = np.random.randint(50, 200)
        start = np.random.randint(0, max(1, len(signal) - burst_len))
        noise[start:start + burst_len] = np.random.normal(0, amplitude, burst_len)
    return signal + noise


def random_lead_inversion(signal: np.ndarray, prob: float = 0.1) -> np.ndarray:
    """Randomly invert the signal (simulates lead reversal)."""
    if np.random.random() < prob:
        return -signal
    return signal


def augment_ecg(
    signal: np.ndarray,
    fs: int,
    target: Optional[np.ndarray] = None,
    prob: float = 0.5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply a random subset of augmentations."""
    aug = signal.copy()

    if np.random.random() < prob:
        aug = amplitude_scale(aug)
    if np.random.random() < prob * 0.5:
        shift = np.random.randint(1, 20)
        aug = temporal_shift(aug, shift)
        if target is not None:
            target = temporal_shift(target, shift)
    if np.random.random() < prob:
        snr = np.random.uniform(10, 30)
        aug = add_gaussian_noise(aug, snr)
    if np.random.random() < prob * 0.3:
        aug = add_baseline_wander(aug, fs, np.random.uniform(0.1, 0.5))
    if np.random.random() < prob * 0.2:
        aug = add_emg_noise(aug, np.random.uniform(0.02, 0.1))

    return aug, target
