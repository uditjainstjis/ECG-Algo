"""
Binary ECG File Parser
========================
Parses the hackathon binary ECG format:
  - 10-byte records: 2-byte int16 (ECG value) + 8-byte int64 LE (timestamp ms)
  - Sequential, no headers, no separators
"""

import numpy as np
import struct
from dataclasses import dataclass
from typing import Tuple


RECORD_SIZE = 10  # bytes per sample


@dataclass
class BinaryECGRecord:
    """Parsed binary ECG file."""
    signal: np.ndarray          # ECG amplitude values (int16 → float64)
    timestamps_ms: np.ndarray   # Timestamps in milliseconds
    fs: int                     # Inferred sampling rate (Hz)
    duration_sec: float         # Total duration in seconds
    n_samples: int              # Number of samples


def parse_binary_ecg(data: bytes) -> BinaryECGRecord:
    """
    Parse binary ECG data per hackathon spec.

    Format per sample (10 bytes):
        [0:2]  int16 LE  — ECG amplitude
        [2:10] int64 LE  — timestamp in milliseconds

    Args:
        data: Raw bytes from the .bin file

    Returns:
        BinaryECGRecord with signal, timestamps, and inferred sample rate
    """
    n_samples = len(data) // RECORD_SIZE
    if n_samples == 0:
        raise ValueError("File is empty or too small")

    # Unpack all records at once using numpy for speed
    # Each record: int16 (2 bytes) + int64 (8 bytes) = 10 bytes
    ecg_values = np.zeros(n_samples, dtype=np.int16)
    timestamps = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        offset = i * RECORD_SIZE
        ecg_values[i] = struct.unpack_from('<h', data, offset)[0]       # int16 LE
        timestamps[i] = struct.unpack_from('<q', data, offset + 2)[0]   # int64 LE

    # Convert ECG to float
    signal = ecg_values.astype(np.float64)

    # Infer sampling rate from timestamp differences
    if n_samples >= 2:
        dt_ms = np.diff(timestamps).astype(np.float64)
        # Filter out outliers (use median for robustness)
        median_dt = np.median(dt_ms)
        if median_dt > 0:
            fs = int(round(1000.0 / median_dt))
        else:
            fs = 250  # fallback
    else:
        fs = 250

    duration_sec = (timestamps[-1] - timestamps[0]) / 1000.0 if n_samples >= 2 else 0.0

    return BinaryECGRecord(
        signal=signal,
        timestamps_ms=timestamps,
        fs=max(fs, 1),
        duration_sec=duration_sec,
        n_samples=n_samples,
    )


def parse_binary_ecg_fast(data: bytes) -> BinaryECGRecord:
    """
    Fast numpy-based parser using structured dtype.
    Much faster for large files.
    """
    n_samples = len(data) // RECORD_SIZE
    if n_samples == 0:
        raise ValueError("File is empty or too small")

    # Define structured dtype matching the binary layout
    dt = np.dtype([('ecg', '<i2'), ('timestamp', '<i8')])
    records = np.frombuffer(data[:n_samples * RECORD_SIZE], dtype=dt)

    signal = records['ecg'].astype(np.float64)
    timestamps = records['timestamp']

    # Infer fs
    if n_samples >= 2:
        dt_ms = np.median(np.diff(timestamps).astype(np.float64))
        fs = int(round(1000.0 / dt_ms)) if dt_ms > 0 else 250
    else:
        fs = 250

    duration_sec = float(timestamps[-1] - timestamps[0]) / 1000.0 if n_samples >= 2 else 0.0

    return BinaryECGRecord(
        signal=signal,
        timestamps_ms=timestamps,
        fs=max(fs, 1),
        duration_sec=duration_sec,
        n_samples=n_samples,
    )
