#!/usr/bin/env python3
"""
HeartEngine Full Pipeline Runner
===================================
Runs all 4 stages on a PhysioNet recording and produces the clinical report.

Usage:
    python scripts/run_full_pipeline.py --dataset afdb --record 04015
"""

import sys
import os
import argparse
import logging
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heartengine.config import CONFIG
from heartengine.data.physionet_loader import load_record, segment_signal
from heartengine.data.preprocessing import preprocess_ecg
from heartengine.stage_a.pan_tompkins import AdaptivePanTompkins
from heartengine.stage_a.signal_quality import compute_sqi
from heartengine.stage_a.hrv_features import extract_hrv_features
from heartengine.ensemble.afib_scanner import scan_for_afib
from heartengine.stage_d.narrative_generator import generate_clinical_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("HeartEngine")


def run_pipeline(dataset_dir: str, record_name: str, target_fs: int = 250):
    """Run the full HeartEngine pipeline on a single record."""

    # ========== LOAD DATA ==========
    logger.info(f"{'='*60}")
    logger.info(f"HeartEngine Pipeline — Record: {record_name}")
    logger.info(f"{'='*60}")

    record_path = os.path.join(dataset_dir, record_name)
    rec = load_record(record_path, target_fs=target_fs)
    logger.info(f"Loaded: {rec.duration_sec:.1f}s, {rec.fs}Hz, lead={rec.lead_name}, "
                f"{len(rec.r_peaks_gold)} gold R-peaks")

    # ========== STAGE A: Classical DSP ==========
    logger.info("\n--- STAGE A: Classical DSP (Pan-Tompkins) ---")

    # Preprocess
    cleaned = preprocess_ecg(rec.signal, rec.fs)

    # R-peak detection (Pan-Tompkins has its own bandpass, use raw signal for
    # consistent peak coordinates with gold standard)
    pt = AdaptivePanTompkins()
    result_a = pt.detect(rec.signal, rec.fs)
    logger.info(f"Pan-Tompkins detected {len(result_a.rpeaks)} R-peaks")

    # Signal Quality Index (5-second windows)
    sqi_scores = []
    window_size = int(5 * rec.fs)
    for i in range(0, len(cleaned) - window_size, window_size):
        window = cleaned[i:i + window_size]
        sqi, _ = compute_sqi(window, rec.fs)
        sqi_scores.append(sqi)
    mean_sqi = np.mean(sqi_scores) if sqi_scores else 0.5
    analyzable_pct = np.mean([s > CONFIG.stage_a.SQI_THRESHOLD for s in sqi_scores]) * 100
    logger.info(f"Signal Quality: mean={mean_sqi:.3f}, analyzable={analyzable_pct:.1f}%")

    # Heart rate statistics
    if len(result_a.heart_rate_bpm) > 0:
        mean_hr = float(np.mean(result_a.heart_rate_bpm))
        min_hr = float(np.min(result_a.heart_rate_bpm))
        max_hr = float(np.max(result_a.heart_rate_bpm))
    else:
        mean_hr, min_hr, max_hr = 0, 0, 0
    logger.info(f"Heart Rate: mean={mean_hr:.1f}, range=[{min_hr:.0f}, {max_hr:.0f}] BPM")

    # HRV features for XGBoost
    if len(result_a.rr_intervals_sec) >= 20:
        hrv = extract_hrv_features(result_a.rr_intervals_sec)
        logger.info(f"HRV: CV={hrv.get('cv_rr', 0):.4f}, RMSSD={hrv.get('rmssd', 0):.4f}, "
                    f"SampEn={hrv.get('sample_entropy', 0):.4f}")

    # ========== STAGE A: AFib Scanning ==========
    logger.info("\n--- AFib Episode Scanning ---")

    if len(result_a.rpeaks) >= 2:
        rr_times = np.cumsum(result_a.rr_intervals_sec)
        episodes, window_details = scan_for_afib(
            result_a.rr_intervals_sec,
            rr_times,
            window_beats=min(100, len(result_a.rr_intervals_sec)),
            stride_beats=min(50, len(result_a.rr_intervals_sec) // 2),
        )
        logger.info(f"Detected {len(episodes)} AF episode(s)")
        for i, ep in enumerate(episodes):
            logger.info(f"  Episode {i+1}: {ep.start_sec:.1f}s – {ep.end_sec:.1f}s "
                        f"({ep.duration_sec:.0f}s), HR={ep.mean_hr:.0f}, CV={ep.rr_variability:.3f}, "
                        f"conf={ep.confidence:.2f}")
    else:
        episodes = []
        window_details = []
        logger.warning("Too few R-peaks for rhythm analysis")

    # ========== EVALUATION vs GOLD STANDARD ==========
    if len(rec.r_peaks_gold) > 0 and len(result_a.rpeaks) > 0:
        logger.info("\n--- R-Peak Detection Evaluation ---")
        tolerance = int(150 * rec.fs / 1000)  # 150ms tolerance (accounts for filter delay)
        tp, fp, fn = _evaluate_peaks(result_a.rpeaks, rec.r_peaks_gold, tolerance)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * sensitivity * ppv / (sensitivity + ppv) if (sensitivity + ppv) > 0 else 0
        logger.info(f"TP={tp}, FP={fp}, FN={fn}")
        logger.info(f"Sensitivity={sensitivity:.4f}, PPV={ppv:.4f}, F1={f1:.4f}")

    # ========== GENERATE REPORT ==========
    logger.info("\n--- Generating Clinical Report ---")

    stage_metrics = {
        "Stage A (Classical)": {
            "method": "Pan-Tompkins + HRV Heuristics",
            "af_detected": len(episodes) > 0,
            "confidence": max([ep.confidence for ep in episodes]) if episodes else 0,
        },
    }

    report = generate_clinical_report(
        recording_name=record_name,
        duration_sec=rec.duration_sec,
        total_beats=len(result_a.rpeaks),
        mean_hr=mean_hr, min_hr=min_hr, max_hr=max_hr,
        afib_episodes=episodes,
        sqi_summary={"analyzable_pct": analyzable_pct, "mean_sqi": mean_sqi},
        stage_metrics=stage_metrics,
        rhythm_summary={},
    )

    # Save report
    report_path = os.path.join(CONFIG.paths.RESULTS_DIR, f"report_{record_name}.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to: {report_path}")

    print("\n" + report)
    return result_a, episodes, report


def _evaluate_peaks(detected: np.ndarray, gold: np.ndarray, tolerance: int):
    """Evaluate detected peaks against gold standard."""
    tp = 0
    used_gold = set()
    for d in detected:
        diffs = np.abs(gold - d)
        j = np.argmin(diffs)
        if diffs[j] <= tolerance and j not in used_gold:
            tp += 1
            used_gold.add(j)
    fp = len(detected) - tp
    fn = len(gold) - tp
    return tp, fp, fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HeartEngine Full Pipeline")
    parser.add_argument("--dataset", default="afdb", help="Dataset name (mitdb, afdb)")
    parser.add_argument("--record", default="04015", help="Record name")
    parser.add_argument("--fs", type=int, default=250, help="Target sampling rate")
    args = parser.parse_args()

    dataset_dir = os.path.join(CONFIG.paths.DATA_DIR, args.dataset)
    if not os.path.exists(dataset_dir):
        logger.info(f"Dataset not found at {dataset_dir}. Downloading...")
        from heartengine.data.download import download_dataset
        download_dataset(args.dataset, CONFIG.paths.DATA_DIR)

    run_pipeline(dataset_dir, args.record, args.fs)
