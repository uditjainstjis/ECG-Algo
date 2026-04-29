"""
PhysioNet Record Loader
========================
Loads WFDB records, extracts single-lead ECG, annotations, and R-peak gold standard.
"""

import os
import numpy as np
import wfdb
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from scipy.signal import resample_poly
from math import gcd

logger = logging.getLogger(__name__)


@dataclass
class ECGRecord:
    """Container for a loaded ECG record."""
    record_name: str
    signal: np.ndarray          # 1D single-lead ECG signal
    fs: int                     # Sampling rate (Hz)
    duration_sec: float         # Duration in seconds
    r_peaks_gold: np.ndarray    # Gold-standard R-peak indices (if available)
    beat_labels: np.ndarray     # Beat type annotations (N, V, A, etc.)
    rhythm_labels: list         # Rhythm annotation spans [(start_idx, end_idx, label), ...]
    lead_name: str              # Which lead was extracted


def load_record(
    record_path: str,
    lead: str = "MLII",
    target_fs: Optional[int] = None,
) -> ECGRecord:
    """
    Load a single WFDB record and extract one lead.

    Args:
        record_path: Full path to record (without extension)
        lead: Preferred lead name (falls back to first channel)
        target_fs: If set, resample to this rate

    Returns:
        ECGRecord with single-lead signal and annotations
    """
    record = wfdb.rdrecord(record_path)
    fs = record.fs

    # Select lead
    lead_names = record.sig_name
    if lead in lead_names:
        lead_idx = lead_names.index(lead)
    else:
        lead_idx = 0
        lead = lead_names[0]

    signal = record.p_signal[:, lead_idx].astype(np.float64)

    # Load annotations if available
    r_peaks_gold = np.array([], dtype=int)
    beat_labels = np.array([], dtype=str)
    rhythm_labels = []

    try:
        ann = wfdb.rdann(record_path, "atr")

        # Beat annotations (R-peak locations)
        beat_types = {"N", "L", "R", "B", "A", "a", "J", "S", "V", "r",
                      "F", "e", "j", "n", "E", "/", "f", "Q", "?"}
        beat_mask = np.array([s in beat_types for s in ann.symbol])
        r_peaks_gold = ann.sample[beat_mask]
        beat_labels = np.array(ann.symbol)[beat_mask]

        # Rhythm annotations
        rhythm_types = {"(N", "(AFIB", "(AFL", "(J", "(AB", "(SVTA", "(VT",
                        "(IVR", "(B", "(T", "(SBR", "(BII", "(NOD", "(P",
                        "(PREX"}
        for i, (samp, sym, aux) in enumerate(zip(ann.sample, ann.symbol, ann.aux_note)):
            aux_clean = aux.strip() if aux else ""
            if sym == "+" and aux_clean:
                # Find end of this rhythm
                end_samp = len(signal)
                for j in range(i + 1, len(ann.sample)):
                    if ann.symbol[j] == "+":
                        end_samp = ann.sample[j]
                        break
                rhythm_labels.append((samp, end_samp, aux_clean))

    except FileNotFoundError:
        logger.warning(f"No annotation file found for {record_path}")
    except Exception as e:
        logger.warning(f"Error reading annotations: {e}")

    # Resample if needed
    original_fs = fs
    if target_fs is not None and target_fs != fs:
        g = gcd(target_fs, fs)
        up, down = target_fs // g, fs // g
        signal = resample_poly(signal, up, down)

        # Adjust peak positions
        if len(r_peaks_gold) > 0:
            r_peaks_gold = np.round(r_peaks_gold * target_fs / original_fs).astype(int)
            r_peaks_gold = np.clip(r_peaks_gold, 0, len(signal) - 1)

        # Adjust rhythm labels
        rhythm_labels = [
            (int(round(s * target_fs / original_fs)),
             int(round(e * target_fs / original_fs)),
             label)
            for s, e, label in rhythm_labels
        ]
        fs = target_fs

    return ECGRecord(
        record_name=os.path.basename(record_path),
        signal=signal,
        fs=fs,
        duration_sec=len(signal) / fs,
        r_peaks_gold=r_peaks_gold,
        beat_labels=beat_labels,
        rhythm_labels=rhythm_labels,
        lead_name=lead,
    )


def segment_signal(
    signal: np.ndarray,
    fs: int,
    window_sec: float,
    overlap: float = 0.5,
    r_peaks: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, int, int, Optional[np.ndarray]]]:
    """
    Segment a long ECG signal into overlapping windows.

    Returns:
        List of (segment, start_idx, end_idx, local_r_peaks) tuples
    """
    window_samples = int(window_sec * fs)
    stride_samples = int(window_samples * (1 - overlap))
    segments = []

    for start in range(0, len(signal) - window_samples + 1, stride_samples):
        end = start + window_samples
        seg = signal[start:end]

        # Extract local R-peaks (relative to window start)
        local_peaks = None
        if r_peaks is not None:
            mask = (r_peaks >= start) & (r_peaks < end)
            local_peaks = r_peaks[mask] - start

        segments.append((seg, start, end, local_peaks))

    return segments


def load_all_records(dataset_dir: str, target_fs: Optional[int] = None) -> List[ECGRecord]:
    """Load all records from a dataset directory."""
    records = []
    dat_files = sorted([f.replace(".dat", "") for f in os.listdir(dataset_dir) if f.endswith(".dat")])

    for rec_name in dat_files:
        try:
            rec = load_record(os.path.join(dataset_dir, rec_name), target_fs=target_fs)
            records.append(rec)
            logger.info(f"Loaded {rec_name}: {rec.duration_sec:.1f}s, {len(rec.r_peaks_gold)} beats")
        except Exception as e:
            logger.warning(f"Failed to load {rec_name}: {e}")

    return records
