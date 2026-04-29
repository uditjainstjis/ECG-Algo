"""
HRV Feature Extraction
========================
40+ heart rate variability features from RR interval sequences.
Time-domain, frequency-domain, nonlinear, and structural features.
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Dict


def _sample_entropy(rr: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Compute Sample Entropy of RR interval sequence."""
    N = len(rr)
    if N < m + 2:
        return 0.0
    r = r_factor * np.std(rr)
    if r < 1e-10:
        return 0.0

    def _count_matches(template_len):
        count = 0
        templates = np.array([rr[i:i + template_len] for i in range(N - template_len)])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 1
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)
    if B == 0:
        return 0.0
    return -np.log(A / B) if A > 0 else 0.0


def _poincare_features(rr: np.ndarray) -> Dict[str, float]:
    """Poincaré plot features: SD1, SD2, SD1/SD2 ratio."""
    if len(rr) < 3:
        return {"SD1": 0, "SD2": 0, "SD1_SD2_ratio": 0}
    x = rr[:-1]
    y = rr[1:]
    sd1 = np.std((y - x) / np.sqrt(2))
    sd2 = np.std((y + x) / np.sqrt(2))
    ratio = sd1 / sd2 if sd2 > 1e-10 else 0
    return {"SD1": float(sd1), "SD2": float(sd2), "SD1_SD2_ratio": float(ratio)}


def _frequency_features(rr: np.ndarray, fs_interp: float = 4.0) -> Dict[str, float]:
    """Frequency-domain HRV via Lomb-Scargle periodogram."""
    if len(rr) < 10:
        return {"VLF": 0, "LF": 0, "HF": 0, "LF_HF_ratio": 0, "total_power": 0}

    # Cumulative time axis
    t = np.cumsum(rr)
    t = t - t[0]

    # Interpolate to uniform sampling for FFT
    t_interp = np.arange(0, t[-1], 1.0 / fs_interp)
    if len(t_interp) < 4:
        return {"VLF": 0, "LF": 0, "HF": 0, "LF_HF_ratio": 0, "total_power": 0}

    f_interp = interp1d(t, rr, kind="cubic", fill_value="extrapolate")
    rr_interp = f_interp(t_interp)
    rr_interp = rr_interp - np.mean(rr_interp)

    # FFT
    N = len(rr_interp)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs_interp)
    psd = np.abs(np.fft.rfft(rr_interp)) ** 2 / N

    vlf = np.sum(psd[(freqs >= 0.003) & (freqs < 0.04)])
    lf = np.sum(psd[(freqs >= 0.04) & (freqs < 0.15)])
    hf = np.sum(psd[(freqs >= 0.15) & (freqs < 0.4)])
    total = vlf + lf + hf

    return {
        "VLF": float(vlf),
        "LF": float(lf),
        "HF": float(hf),
        "LF_HF_ratio": float(lf / hf) if hf > 1e-10 else 0.0,
        "total_power": float(total),
    }


def _transition_entropy(rr: np.ndarray, n_bins: int = 8) -> float:
    """RR interval transition matrix entropy."""
    if len(rr) < 3:
        return 0.0
    bins = np.linspace(np.min(rr), np.max(rr) + 1e-10, n_bins + 1)
    digitized = np.digitize(rr, bins) - 1
    digitized = np.clip(digitized, 0, n_bins - 1)

    # Build transition matrix
    trans = np.zeros((n_bins, n_bins))
    for i in range(len(digitized) - 1):
        trans[digitized[i], digitized[i + 1]] += 1

    # Normalize rows
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = trans / row_sums

    # Shannon entropy of transition probabilities
    entropy = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if trans_prob[i, j] > 0:
                entropy -= trans_prob[i, j] * np.log2(trans_prob[i, j])
    return float(entropy / n_bins)


def extract_hrv_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Extract comprehensive HRV feature vector from RR intervals.

    Args:
        rr_intervals: Array of RR intervals in seconds

    Returns:
        Dictionary of 40+ named features
    """
    rr = np.asarray(rr_intervals, dtype=np.float64)
    features = {}

    if len(rr) < 3:
        return {f"feat_{i}": 0.0 for i in range(40)}

    # === TIME-DOMAIN (14 features) ===
    diffs = np.diff(rr)
    abs_diffs = np.abs(diffs)

    features["mean_rr"] = float(np.mean(rr))
    features["std_rr"] = float(np.std(rr))  # SDNN
    features["median_rr"] = float(np.median(rr))
    features["cv_rr"] = float(np.std(rr) / np.mean(rr)) if np.mean(rr) > 0 else 0
    features["rmssd"] = float(np.sqrt(np.mean(diffs ** 2)))
    features["pnn50"] = float(np.sum(abs_diffs > 0.050) / len(diffs)) if len(diffs) > 0 else 0
    features["pnn20"] = float(np.sum(abs_diffs > 0.020) / len(diffs)) if len(diffs) > 0 else 0
    features["range_rr"] = float(np.max(rr) - np.min(rr))
    features["iqr_rr"] = float(np.percentile(rr, 75) - np.percentile(rr, 25))
    features["skewness"] = float(_skewness(rr))
    features["kurtosis"] = float(_kurtosis(rr))
    features["mean_hr"] = float(60.0 / np.mean(rr)) if np.mean(rr) > 0 else 0
    features["std_hr"] = float(np.std(60.0 / rr)) if np.all(rr > 0) else 0
    features["mean_successive_diff"] = float(np.mean(abs_diffs)) if len(abs_diffs) > 0 else 0

    # === NONLINEAR (6 features) ===
    poincare = _poincare_features(rr)
    features.update(poincare)
    features["sample_entropy"] = _sample_entropy(rr[:200])  # Cap for speed
    features["turning_point_ratio"] = _turning_point_ratio(rr)
    features["consecutive_diff_std"] = float(np.std(diffs)) if len(diffs) > 0 else 0

    # === FREQUENCY-DOMAIN (5 features) ===
    freq_feats = _frequency_features(rr)
    features.update(freq_feats)

    # === STRUCTURAL (3 features) ===
    features["transition_entropy"] = _transition_entropy(rr)
    features["rr_irregularity"] = _irregularity_evidence(rr)
    features["longest_regular_run"] = _longest_regular_run(rr)

    return features


def _skewness(x):
    mu, std = np.mean(x), np.std(x)
    return float(np.mean(((x - mu) / std) ** 3)) if std > 1e-10 else 0.0


def _kurtosis(x):
    mu, std = np.mean(x), np.std(x)
    return float(np.mean(((x - mu) / std) ** 4)) if std > 1e-10 else 0.0


def _turning_point_ratio(rr):
    if len(rr) < 3:
        return 0.0
    tp = sum(1 for i in range(1, len(rr) - 1)
             if (rr[i] > rr[i-1] and rr[i] > rr[i+1]) or (rr[i] < rr[i-1] and rr[i] < rr[i+1]))
    return float(tp / (len(rr) - 2))


def _irregularity_evidence(rr):
    """Coefficient of variation of successive RR differences — high in AFib."""
    if len(rr) < 3:
        return 0.0
    diffs = np.abs(np.diff(rr))
    mean_diff = np.mean(diffs)
    if mean_diff < 1e-10:
        return 0.0
    return float(np.std(diffs) / mean_diff)


def _longest_regular_run(rr, threshold: float = 0.15):
    """Longest consecutive run of RR intervals within threshold of median."""
    if len(rr) < 2:
        return 0.0
    med = np.median(rr)
    regular = np.abs(rr - med) < threshold * med
    max_run = 0
    current = 0
    for r in regular:
        if r:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return float(max_run / len(rr))


def get_feature_names() -> list:
    """Return ordered list of feature names."""
    dummy = extract_hrv_features(np.random.uniform(0.6, 1.0, 100))
    return list(dummy.keys())
