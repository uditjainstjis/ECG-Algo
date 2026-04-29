"""
AFib Episode Scanner
======================
Continuously scans long ECG recordings for AF episodes.
Combines per-window AF classification with temporal smoothing,
episode merging, and confidence-based reporting.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ..stage_a.hrv_features import extract_hrv_features


@dataclass
class AFibEpisode:
    """Detected AF episode."""
    start_sec: float
    end_sec: float
    duration_sec: float
    confidence: float
    mean_hr: float
    rr_variability: float  # CV of RR intervals
    num_windows: int


def compute_window_irregularity(rr_intervals: np.ndarray) -> dict:
    """
    Compute irregularity metrics for a single window of RR intervals.

    Returns dict with key AF-indicative features:
    - cv_rr: Coefficient of variation (high in AF)
    - rmssd: Root mean square of successive differences
    - turning_point_ratio: Proportion of turning points
    - sample_entropy: Complexity measure
    """
    if len(rr_intervals) < 5:
        return {"cv_rr": 0, "rmssd": 0, "turning_point_ratio": 0, "sample_entropy": 0, "is_irregular": False}

    feats = extract_hrv_features(rr_intervals)

    # Heuristic AF criteria:
    # 1. High RR variability (CV > 0.15)
    # 2. High RMSSD (> 0.1s)
    # 3. High turning point ratio (close to 2/3 = maximally irregular)
    cv = feats.get("cv_rr", 0)
    rmssd = feats.get("rmssd", 0)
    tpr = feats.get("turning_point_ratio", 0)
    samp_en = feats.get("sample_entropy", 0)

    af_score = (
        0.35 * min(cv / 0.25, 1.0) +
        0.25 * min(rmssd / 0.15, 1.0) +
        0.20 * min(tpr / 0.667, 1.0) +
        0.20 * min(samp_en / 2.0, 1.0)
    )

    return {
        "cv_rr": cv,
        "rmssd": rmssd,
        "turning_point_ratio": tpr,
        "sample_entropy": samp_en,
        "af_heuristic_score": af_score,
        "is_irregular": af_score > 0.5,
    }


def scan_for_afib(
    rr_intervals: np.ndarray,
    rr_times: np.ndarray,
    window_beats: int = 100,
    stride_beats: int = 50,
    af_threshold: float = 0.5,
    merge_gap_sec: float = 30.0,
    min_episode_sec: float = 10.0,
    ml_predictions: Optional[np.ndarray] = None,
    ml_weight: float = 0.6,
) -> Tuple[List[AFibEpisode], List[dict]]:
    """
    Scan a full recording for AF episodes.

    Args:
        rr_intervals: Full sequence of RR intervals (seconds)
        rr_times: Cumulative time of each R-peak (seconds)
        window_beats: Number of RR intervals per analysis window
        stride_beats: Stride in beats
        af_threshold: AF probability threshold
        merge_gap_sec: Merge episodes separated by less than this
        min_episode_sec: Minimum episode duration to report
        ml_predictions: Optional per-window ML predictions (from Stage B/C)
        ml_weight: Weight for ML predictions vs heuristic

    Returns:
        (episodes, window_details)
    """
    n = len(rr_intervals)
    window_results = []

    for start in range(0, n - window_beats + 1, stride_beats):
        end = start + window_beats
        window_rr = rr_intervals[start:end]

        # Time boundaries
        t_start = float(rr_times[start]) if start < len(rr_times) else 0
        t_end = float(rr_times[min(end, len(rr_times) - 1)])

        # Heuristic analysis
        irregularity = compute_window_irregularity(window_rr)

        # Combine with ML prediction if available
        window_idx = len(window_results)
        if ml_predictions is not None and window_idx < len(ml_predictions):
            ml_prob = float(ml_predictions[window_idx])
            heuristic_weight = 1.0 - ml_weight
            combined_prob = ml_weight * ml_prob + heuristic_weight * irregularity["af_heuristic_score"]
        else:
            combined_prob = irregularity["af_heuristic_score"]

        window_results.append({
            "start_sec": t_start,
            "end_sec": t_end,
            "start_beat": start,
            "end_beat": end,
            "af_probability": combined_prob,
            "is_af": combined_prob >= af_threshold,
            "mean_hr": float(60.0 / np.mean(window_rr)) if np.mean(window_rr) > 0 else 0,
            **irregularity,
        })

    # Temporal smoothing: median filter over 5 windows
    probs = np.array([w["af_probability"] for w in window_results])
    if len(probs) >= 5:
        from scipy.ndimage import median_filter
        smoothed = median_filter(probs, size=5)
        for i, w in enumerate(window_results):
            w["af_probability_smoothed"] = float(smoothed[i])
            w["is_af"] = smoothed[i] >= af_threshold
    else:
        for w in window_results:
            w["af_probability_smoothed"] = w["af_probability"]

    # Detect episodes (contiguous AF windows)
    episodes = []
    in_episode = False
    ep_start = 0

    for i, w in enumerate(window_results):
        if w["is_af"] and not in_episode:
            in_episode = True
            ep_start = i
        elif not w["is_af"] and in_episode:
            in_episode = False
            _add_episode(episodes, window_results, ep_start, i - 1)

    if in_episode:
        _add_episode(episodes, window_results, ep_start, len(window_results) - 1)

    # Merge nearby episodes
    merged = _merge_episodes(episodes, merge_gap_sec)

    # Filter by minimum duration
    merged = [ep for ep in merged if ep.duration_sec >= min_episode_sec]

    return merged, window_results


def _add_episode(episodes, windows, start_idx, end_idx):
    ep_windows = windows[start_idx:end_idx + 1]
    episodes.append(AFibEpisode(
        start_sec=ep_windows[0]["start_sec"],
        end_sec=ep_windows[-1]["end_sec"],
        duration_sec=ep_windows[-1]["end_sec"] - ep_windows[0]["start_sec"],
        confidence=float(np.mean([w["af_probability_smoothed"] for w in ep_windows])),
        mean_hr=float(np.mean([w["mean_hr"] for w in ep_windows])),
        rr_variability=float(np.mean([w["cv_rr"] for w in ep_windows])),
        num_windows=len(ep_windows),
    ))


def _merge_episodes(episodes: List[AFibEpisode], gap_sec: float) -> List[AFibEpisode]:
    if len(episodes) <= 1:
        return episodes

    merged = [episodes[0]]
    for ep in episodes[1:]:
        if ep.start_sec - merged[-1].end_sec <= gap_sec:
            # Merge
            prev = merged[-1]
            merged[-1] = AFibEpisode(
                start_sec=prev.start_sec,
                end_sec=ep.end_sec,
                duration_sec=ep.end_sec - prev.start_sec,
                confidence=(prev.confidence + ep.confidence) / 2,
                mean_hr=(prev.mean_hr + ep.mean_hr) / 2,
                rr_variability=(prev.rr_variability + ep.rr_variability) / 2,
                num_windows=prev.num_windows + ep.num_windows,
            )
        else:
            merged.append(ep)

    return merged
