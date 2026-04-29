"""
Multi-Detector R-Peak Consensus
==================================
Novel contribution: weighted voting across Classical DSP and Deep Learning
R-peak detectors, with weights conditioned on Signal Quality Index.
"""

import numpy as np
from typing import List, Tuple, Optional


def match_peaks(
    peaks_a: np.ndarray,
    peaks_b: np.ndarray,
    tolerance_samples: int = 20,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match peaks between two detectors within tolerance.

    Returns:
        (matched_pairs, unmatched_a, unmatched_b)
    """
    matched = []
    used_b = set()

    for i, pa in enumerate(peaks_a):
        if len(peaks_b) == 0:
            break
        diffs = np.abs(peaks_b - pa)
        j = int(np.argmin(diffs))
        if diffs[j] <= tolerance_samples and j not in used_b:
            matched.append((i, j))
            used_b.add(j)

    unmatched_a = [i for i in range(len(peaks_a)) if i not in {m[0] for m in matched}]
    unmatched_b = [j for j in range(len(peaks_b)) if j not in used_b]

    return matched, unmatched_a, unmatched_b


def rpeak_consensus(
    peaks_list: List[np.ndarray],
    confidence_list: List[np.ndarray],
    weights: List[float],
    tolerance_samples: int = 20,
    consensus_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-detector R-peak consensus via weighted voting.

    Args:
        peaks_list: List of R-peak arrays from each detector
        confidence_list: Confidence scores per peak per detector
        weights: Detector weights (e.g., conditioned on SQI)
        tolerance_samples: Max distance for peak matching
        consensus_threshold: Minimum weighted confidence to accept

    Returns:
        (consensus_peaks, consensus_confidences)
    """
    if len(peaks_list) == 0:
        return np.array([], dtype=int), np.array([])

    if len(peaks_list) == 1:
        return peaks_list[0], confidence_list[0]

    # Collect all candidate peaks
    all_peaks = []
    for det_idx, (peaks, confs) in enumerate(zip(peaks_list, confidence_list)):
        for i, p in enumerate(peaks):
            c = confs[i] if i < len(confs) else 0.5
            all_peaks.append({"pos": int(p), "conf": float(c), "det": det_idx})

    if not all_peaks:
        return np.array([], dtype=int), np.array([])

    # Sort by position
    all_peaks.sort(key=lambda x: x["pos"])

    # Cluster nearby peaks
    clusters = []
    current_cluster = [all_peaks[0]]

    for p in all_peaks[1:]:
        if p["pos"] - current_cluster[-1]["pos"] <= tolerance_samples:
            current_cluster.append(p)
        else:
            clusters.append(current_cluster)
            current_cluster = [p]
    clusters.append(current_cluster)

    # Vote within each cluster
    consensus_peaks = []
    consensus_confs = []

    for cluster in clusters:
        # Weighted confidence across detectors
        total_weight = 0.0
        weighted_conf = 0.0
        weighted_pos = 0.0
        detectors_present = set()

        for p in cluster:
            w = weights[p["det"]] if p["det"] < len(weights) else 1.0
            weighted_conf += w * p["conf"]
            weighted_pos += w * p["pos"]
            total_weight += w
            detectors_present.add(p["det"])

        if total_weight > 0:
            avg_conf = weighted_conf / total_weight
            avg_pos = int(round(weighted_pos / total_weight))

            # Boost confidence for multi-detector agreement
            agreement_bonus = len(detectors_present) / len(peaks_list)
            final_conf = avg_conf * (0.7 + 0.3 * agreement_bonus)

            if final_conf >= consensus_threshold:
                consensus_peaks.append(avg_pos)
                consensus_confs.append(min(final_conf, 1.0))

    return np.array(consensus_peaks, dtype=int), np.array(consensus_confs)


def sqi_weighted_fusion(
    sqi_score: float,
    classical_peaks: np.ndarray,
    dl_peaks: np.ndarray,
    classical_conf: Optional[np.ndarray] = None,
    dl_conf: Optional[np.ndarray] = None,
    tolerance_samples: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SQI-conditioned ensemble:
    - Low SQI → trust classical (more robust to noise)
    - High SQI → trust deep learning (higher precision)
    """
    if classical_conf is None:
        classical_conf = np.ones(len(classical_peaks)) * 0.8
    if dl_conf is None:
        dl_conf = np.ones(len(dl_peaks)) * 0.9

    # Weight schedule: sigmoid centered at SQI=0.5
    dl_weight = 1.0 / (1.0 + np.exp(-10 * (sqi_score - 0.5)))
    classical_weight = 1.0 - dl_weight * 0.5  # Classical always has baseline weight

    return rpeak_consensus(
        [classical_peaks, dl_peaks],
        [classical_conf, dl_conf],
        [classical_weight, dl_weight],
        tolerance_samples=tolerance_samples,
    )
