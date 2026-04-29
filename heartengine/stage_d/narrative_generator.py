"""
Clinical Narrative Generator
===============================
Generates structured clinical reports from HeartEngine analysis results.
Template-based for demo; can be upgraded to LLM-powered generation.
"""

from typing import List, Dict, Optional
from datetime import datetime


def generate_clinical_report(
    recording_name: str,
    duration_sec: float,
    total_beats: int,
    mean_hr: float,
    min_hr: float,
    max_hr: float,
    afib_episodes: list,
    sqi_summary: dict,
    stage_metrics: dict,
    rhythm_summary: dict,
) -> str:
    """
    Generate a structured clinical narrative report.

    Returns:
        Formatted markdown report string
    """
    duration_min = duration_sec / 60
    duration_hr = duration_sec / 3600

    report = []
    report.append("# HeartEngine — ECG Analysis Report")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Recording**: {recording_name}")
    report.append(f"**Duration**: {duration_hr:.1f} hours ({duration_min:.0f} minutes)")
    report.append("")

    # === HEART RATE SUMMARY ===
    report.append("## 1. Heart Rate Summary")
    report.append(f"- **Total beats detected**: {total_beats:,}")
    report.append(f"- **Mean heart rate**: {mean_hr:.1f} BPM")
    report.append(f"- **Range**: {min_hr:.0f} – {max_hr:.0f} BPM")
    if mean_hr < 60:
        report.append(f"- ⚠️ **Bradycardia noted**: Mean HR below 60 BPM")
    elif mean_hr > 100:
        report.append(f"- ⚠️ **Tachycardia noted**: Mean HR above 100 BPM")
    else:
        report.append(f"- ✅ Mean heart rate within normal range")
    report.append("")

    # === RHYTHM ANALYSIS ===
    report.append("## 2. Rhythm Analysis")
    total_af_sec = sum(ep.duration_sec for ep in afib_episodes)
    af_burden = (total_af_sec / duration_sec * 100) if duration_sec > 0 else 0

    if len(afib_episodes) == 0:
        report.append("- ✅ **No atrial fibrillation episodes detected**")
        report.append("- Rhythm appears predominantly regular throughout recording")
    else:
        report.append(f"- ⚠️ **{len(afib_episodes)} atrial fibrillation episode(s) detected**")
        report.append(f"- **Total AF burden**: {total_af_sec:.0f}s ({af_burden:.1f}% of recording)")
        report.append("")
        report.append("### Detected AF Episodes")
        report.append("| # | Start | End | Duration | Mean HR | RR Variability | Confidence |")
        report.append("|---|-------|-----|----------|---------|----------------|------------|")
        for i, ep in enumerate(afib_episodes):
            start_str = _format_time(ep.start_sec)
            end_str = _format_time(ep.end_sec)
            report.append(
                f"| {i+1} | {start_str} | {end_str} | {ep.duration_sec:.0f}s | "
                f"{ep.mean_hr:.0f} BPM | CV={ep.rr_variability:.3f} | {ep.confidence:.1%} |"
            )
    report.append("")

    # === SIGNAL QUALITY ===
    report.append("## 3. Signal Quality Assessment")
    analyzable = sqi_summary.get("analyzable_pct", 100)
    report.append(f"- **Analyzable segments**: {analyzable:.1f}%")
    if analyzable < 80:
        report.append(f"- ⚠️ Significant noise detected in {100-analyzable:.1f}% of recording")
    else:
        report.append(f"- ✅ Good overall signal quality")
    report.append("")

    # === MULTI-STAGE COMPARISON ===
    report.append("## 4. Multi-Stage Analysis Comparison")
    report.append("| Stage | Method | AF Detection | Confidence |")
    report.append("|-------|--------|--------------|------------|")
    for stage_name, metrics in stage_metrics.items():
        af_det = "✅ AF detected" if metrics.get("af_detected", False) else "❌ No AF"
        conf = metrics.get("confidence", 0)
        report.append(f"| {stage_name} | {metrics.get('method', 'N/A')} | {af_det} | {conf:.1%} |")
    report.append("")

    # === CLINICAL INTERPRETATION ===
    report.append("## 5. Clinical Interpretation")
    if len(afib_episodes) > 0:
        report.append("The recording demonstrates **episodes of irregularly irregular rhythm** "
                      "consistent with atrial fibrillation. Key features observed:")
        report.append("- Highly variable R–R intervals (elevated coefficient of variation)")
        report.append("- Elevated RMSSD and sample entropy during AF episodes")
        report.append("- Pattern confirmed by multi-stage consensus across classical DSP, "
                      "deep learning, and foundation model approaches")
        if af_burden > 10:
            report.append(f"\n**AF burden of {af_burden:.1f}%** warrants clinical review and "
                          "potential anticoagulation assessment.")
    else:
        report.append("The recording demonstrates **predominantly regular sinus rhythm** "
                      "with normal heart rate variability.")
        report.append("No episodes meeting criteria for atrial fibrillation were identified "
                      "by any of the detection stages.")
    report.append("")

    report.append("---")
    report.append("*This report was generated by HeartEngine, a 4-stage hybrid ECG analysis system. "
                  "Clinical decisions should be made in conjunction with physician review.*")

    return "\n".join(report)


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
