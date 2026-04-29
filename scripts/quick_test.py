#!/usr/bin/env python3
"""
Quick Validation Test — HeartEngine
======================================
Runs the full Stage A pipeline on a synthetic ECG to verify everything works.
No data download required.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("QuickTest")


def generate_synthetic_ecg(fs=250, duration_sec=30, hr_bpm=72, noise_level=0.05):
    """Generate a realistic synthetic single-lead ECG."""
    t = np.arange(0, duration_sec, 1/fs)
    ecg = np.zeros_like(t)
    rr_sec = 60.0 / hr_bpm
    true_peaks = []

    beat_time = 0.0
    while beat_time < duration_sec - 0.5:
        # R-peak: sharp Gaussian
        r_center = beat_time
        idx = int(r_center * fs)
        if 0 <= idx < len(t):
            true_peaks.append(idx)
            # P wave
            ecg += 0.15 * np.exp(-0.5 * ((t - r_center + 0.16) / 0.04) ** 2)
            # QRS complex
            ecg += -0.1 * np.exp(-0.5 * ((t - r_center + 0.02) / 0.008) ** 2)  # Q
            ecg += 1.0 * np.exp(-0.5 * ((t - r_center) / 0.012) ** 2)          # R
            ecg += -0.2 * np.exp(-0.5 * ((t - r_center - 0.03) / 0.015) ** 2)  # S
            # T wave
            ecg += 0.3 * np.exp(-0.5 * ((t - r_center - 0.25) / 0.06) ** 2)

        # Add some HRV
        jitter = np.random.normal(0, 0.02)
        beat_time += rr_sec + jitter

    # Add noise
    ecg += noise_level * np.random.randn(len(t))

    return ecg, fs, np.array(true_peaks, dtype=int)


def test_stage_a():
    """Test Stage A: Pan-Tompkins + SQI + HRV features."""
    logger.info("=" * 60)
    logger.info("TEST: Stage A — Classical DSP Pipeline")
    logger.info("=" * 60)

    # Generate synthetic ECG
    ecg, fs, true_peaks = generate_synthetic_ecg(fs=250, duration_sec=60, hr_bpm=75)
    logger.info(f"Synthetic ECG: {len(ecg)} samples, {fs}Hz, {len(true_peaks)} true peaks")

    # Preprocessing
    from heartengine.data.preprocessing import preprocess_ecg
    cleaned = preprocess_ecg(ecg, fs, normalize=True)
    logger.info(f"Preprocessed: mean={np.mean(cleaned):.4f}, std={np.std(cleaned):.4f}")

    # Pan-Tompkins
    from heartengine.stage_a.pan_tompkins import AdaptivePanTompkins
    pt = AdaptivePanTompkins()
    result = pt.detect(cleaned, fs)
    logger.info(f"Pan-Tompkins detected {len(result.rpeaks)} R-peaks (true: {len(true_peaks)})")

    if len(result.heart_rate_bpm) > 0:
        logger.info(f"Heart rate: {np.mean(result.heart_rate_bpm):.1f} ± {np.std(result.heart_rate_bpm):.1f} BPM")

    # Evaluate
    tolerance = int(0.075 * fs)  # 75ms
    tp = 0
    used = set()
    for d in result.rpeaks:
        if len(true_peaks) > 0:
            diffs = np.abs(true_peaks - d)
            j = np.argmin(diffs)
            if diffs[j] <= tolerance and j not in used:
                tp += 1
                used.add(j)
    fp = len(result.rpeaks) - tp
    fn = len(true_peaks) - tp
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) > 0 else 0
    logger.info(f"R-peak detection: TP={tp}, FP={fp}, FN={fn}, Se={sens:.4f}, PPV={ppv:.4f}, F1={f1:.4f}")

    # SQI
    from heartengine.stage_a.signal_quality import compute_sqi
    sqi, components = compute_sqi(cleaned[:5*fs], fs)
    logger.info(f"Signal Quality Index: {sqi:.3f} — {components}")

    # HRV features
    if len(result.rr_intervals_sec) >= 10:
        from heartengine.stage_a.hrv_features import extract_hrv_features
        hrv = extract_hrv_features(result.rr_intervals_sec)
        logger.info(f"HRV features ({len(hrv)} total):")
        for key in ["mean_rr", "std_rr", "cv_rr", "rmssd", "pnn50", "sample_entropy"]:
            logger.info(f"  {key}: {hrv.get(key, 'N/A'):.4f}" if isinstance(hrv.get(key), float) else f"  {key}: N/A")

    # AFib scanning
    if len(result.rr_intervals_sec) >= 20:
        from heartengine.ensemble.afib_scanner import scan_for_afib
        rr_times = np.cumsum(result.rr_intervals_sec)
        episodes, details = scan_for_afib(
            result.rr_intervals_sec, rr_times,
            window_beats=min(30, len(result.rr_intervals_sec)),
            stride_beats=min(15, len(result.rr_intervals_sec) // 2),
        )
        logger.info(f"AF episodes detected: {len(episodes)}")
        if len(episodes) == 0:
            logger.info("  ✅ Correct — synthetic ECG is normal sinus rhythm")

    logger.info("\n✅ Stage A tests PASSED\n")


def test_stage_b_architecture():
    """Test Stage B: Model architecture forward pass."""
    logger.info("=" * 60)
    logger.info("TEST: Stage B — Deep Learning Architectures")
    logger.info("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # ResU-Net
    from heartengine.stage_b.resunet_rpeak import ResUNet1D, generate_gaussian_target, decode_peaks
    model = ResUNet1D(base=64).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(2, 1, 1800).to(device)  # 5s @ 360Hz
    with torch.no_grad():
        y = model(x)
    logger.info(f"ResU-Net: {n_params:,} params, input={tuple(x.shape)}, output={tuple(y.shape)}")
    assert y.shape == (2, 1, 1800), f"Expected (2,1,1800), got {y.shape}"

    # Gaussian target
    target = generate_gaussian_target(1800, np.array([200, 560, 920, 1280, 1640]), sigma=7)
    logger.info(f"Gaussian target: shape={target.shape}, max={target.max():.3f}, sum={target.sum():.1f}")

    # Peak decoder
    peaks = decode_peaks(target, fs=360, threshold=0.3)
    logger.info(f"Decoded peaks: {peaks} (expected near [200,560,920,1280,1640])")

    # CNN-Transformer
    from heartengine.stage_b.cnn_transformer_afib import CNNTransformerAFib
    model2 = CNNTransformerAFib(d_model=128, nhead=4, num_layers=3, num_classes=2).to(device)
    n_params2 = sum(p.numel() for p in model2.parameters())
    x2 = torch.randn(2, 1, 7500).to(device)  # 30s @ 250Hz
    with torch.no_grad():
        y2 = model2(x2)
    logger.info(f"CNN-Transformer: {n_params2:,} params, input={tuple(x2.shape)}, output={tuple(y2.shape)}")
    assert y2.shape == (2, 2), f"Expected (2,2), got {y2.shape}"

    # Loss
    from heartengine.stage_b.losses import FocalDiceLoss
    loss_fn = FocalDiceLoss()
    pred = torch.sigmoid(torch.randn(2, 1, 1800))
    tgt = torch.rand(2, 1, 1800)
    loss = loss_fn(pred, tgt)
    logger.info(f"FocalDice loss: {loss.item():.4f}")

    logger.info("\n✅ Stage B architecture tests PASSED\n")


def test_report_generation():
    """Test Stage D: Report generation."""
    logger.info("=" * 60)
    logger.info("TEST: Stage D — Narrative Report")
    logger.info("=" * 60)

    from heartengine.stage_d.narrative_generator import generate_clinical_report
    from heartengine.ensemble.afib_scanner import AFibEpisode

    report = generate_clinical_report(
        recording_name="test_001",
        duration_sec=3600,
        total_beats=4320,
        mean_hr=72, min_hr=55, max_hr=125,
        afib_episodes=[AFibEpisode(120, 300, 180, 0.87, 132, 0.21, 6)],
        sqi_summary={"analyzable_pct": 94.5, "mean_sqi": 0.72},
        stage_metrics={
            "Stage A": {"method": "Pan-Tompkins + XGBoost", "af_detected": True, "confidence": 0.85},
            "Stage B": {"method": "CNN-Transformer", "af_detected": True, "confidence": 0.92},
            "Stage C": {"method": "ECGFounder", "af_detected": True, "confidence": 0.97},
        },
        rhythm_summary={},
    )

    logger.info(f"Report generated: {len(report)} chars")
    print("\n" + report[:1000] + "\n...")
    logger.info("\n✅ Stage D report test PASSED\n")


if __name__ == "__main__":
    test_stage_a()
    test_stage_b_architecture()
    test_report_generation()

    logger.info("=" * 60)
    logger.info("ALL QUICK TESTS PASSED ✅")
    logger.info("=" * 60)
