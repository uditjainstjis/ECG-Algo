"""
Pan-Tompkins QRS / R-Peak Detector — From Scratch
====================================================
Faithful implementation of the original 1985 Pan-Tompkins algorithm with:
- Exact 200Hz integer-coefficient bandpass (LP + HP cascade)
- 5-point derivative filter
- Point-wise squaring
- 150ms moving window integrator
- Dual adaptive thresholds with EMA updates
- T-wave discrimination via slope comparison
- 200ms refractory period
- RR-based searchback at 166% of average RR

Also includes an adaptive multi-rate wrapper that auto-resamples to 200Hz,
runs detection, and maps peaks back to the original sampling rate.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class PanTompkinsConfig:
    """Configuration for the Pan-Tompkins detector."""
    fs: float = 200.0
    integration_window_ms: float = 150.0
    refractory_ms: float = 200.0
    t_wave_window_ms: float = 360.0
    searchback_rr_factor: float = 1.66
    rr_low_factor: float = 0.92
    rr_high_factor: float = 1.16
    threshold_weight: float = 0.25
    peak_ema_alpha: float = 0.125
    init_learning_seconds: float = 2.0


@dataclass
class DetectionResult:
    """Result container with full debug traces."""
    rpeaks: np.ndarray
    bandpassed: np.ndarray
    derivative: np.ndarray
    squared: np.ndarray
    integrated: np.ndarray
    peak_indices: np.ndarray
    heart_rate_bpm: np.ndarray
    rr_intervals_sec: np.ndarray
    debug: Dict[str, Any] = field(default_factory=dict)


class PanTompkinsDetector:
    """
    Original Pan-Tompkins QRS detector with exact 200Hz integer filters.

    The algorithm processes ECG signals through a cascade of:
    1. Low-pass filter: H(z) = (1-z^{-6})^2 / (1-z^{-1})^2
    2. High-pass filter: H(z) = (-1/32 + z^{-16} - z^{-17} + z^{-32}/32) / (1-z^{-1})
    3. 5-point derivative
    4. Point-wise squaring
    5. Moving window integrator (150ms = 30 samples @ 200Hz)
    6. Adaptive dual-threshold peak detection with searchback
    """

    def __init__(self, config: Optional[PanTompkinsConfig] = None):
        self.cfg = config or PanTompkinsConfig()

        self.int_win = max(1, int(round(self.cfg.integration_window_ms * self.cfg.fs / 1000.0)))
        self.refractory = int(round(self.cfg.refractory_ms * self.cfg.fs / 1000.0))
        self.t_wave_window = int(round(self.cfg.t_wave_window_ms * self.cfg.fs / 1000.0))
        self.learn_n = int(round(self.cfg.init_learning_seconds * self.cfg.fs))

    @staticmethod
    def _as_float(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).squeeze()
        if x.ndim != 1:
            raise ValueError("ECG input must be 1D.")
        return x

    def lowpass_filter(self, x: np.ndarray) -> np.ndarray:
        """
        Original Pan-Tompkins low-pass filter (200Hz).
        y[n] = 2y[n-1] - y[n-2] + x[n] - 2x[n-6] + x[n-12]
        """
        x = self._as_float(x)
        N = len(x)
        y = np.zeros(N)
        for n in range(N):
            y[n] = (
                (2.0 * y[n - 1] if n >= 1 else 0.0)
                - (y[n - 2] if n >= 2 else 0.0)
                + x[n]
                - (2.0 * x[n - 6] if n >= 6 else 0.0)
                + (x[n - 12] if n >= 12 else 0.0)
            )
        return y

    def highpass_filter(self, x: np.ndarray) -> np.ndarray:
        """
        Original Pan-Tompkins high-pass filter (200Hz).
        y[n] = y[n-1] - x[n]/32 + x[n-16] - x[n-17] + x[n-32]/32
        """
        x = self._as_float(x)
        N = len(x)
        y = np.zeros(N)
        for n in range(N):
            y[n] = (
                (y[n - 1] if n >= 1 else 0.0)
                - x[n] / 32.0
                + (x[n - 16] if n >= 16 else 0.0)
                - (x[n - 17] if n >= 17 else 0.0)
                + ((x[n - 32] if n >= 32 else 0.0) / 32.0)
            )
        return y

    def derivative_filter(self, x: np.ndarray) -> np.ndarray:
        """
        5-point causal derivative.
        y[n] = (1/8)[-x[n-4] - 2x[n-3] + 2x[n-1] + x[n]]
        """
        x = self._as_float(x)
        N = len(x)
        y = np.zeros(N)
        for n in range(N):
            y[n] = (
                -(x[n - 4] if n >= 4 else 0.0)
                - 2.0 * (x[n - 3] if n >= 3 else 0.0)
                + 2.0 * (x[n - 1] if n >= 1 else 0.0)
                + x[n]
            ) / 8.0
        return y

    def moving_window_integrator(self, x: np.ndarray) -> np.ndarray:
        """Moving average over integration_window_ms."""
        x = self._as_float(x)
        w = self.int_win
        cumsum = np.cumsum(np.insert(x, 0, 0.0))
        mwi_vals = (cumsum[w:] - cumsum[:-w]) / w
        out = np.zeros_like(x)
        out[w - 1:] = mwi_vals
        return out

    @staticmethod
    def _find_peaks(sig: np.ndarray) -> np.ndarray:
        """Find local maxima in signal."""
        if len(sig) < 3:
            return np.array([], dtype=int)
        mask = (sig[1:-1] > sig[:-2]) & (sig[1:-1] >= sig[2:])
        return np.where(mask)[0] + 1

    @staticmethod
    def _search_local_peak(sig: np.ndarray, center: int, radius: int) -> int:
        """Find the maximum within ±radius of center."""
        lo = max(0, center - radius)
        hi = min(len(sig), center + radius + 1)
        if lo >= hi:
            return center
        return lo + int(np.argmax(sig[lo:hi]))

    def detect(self, ecg: np.ndarray) -> DetectionResult:
        """
        Run the full Pan-Tompkins detection pipeline.

        Args:
            ecg: 1D ECG signal sampled at self.cfg.fs Hz

        Returns:
            DetectionResult with R-peak indices, HR, RR intervals, and debug info
        """
        ecg = self._as_float(ecg)

        # === SIGNAL PROCESSING CASCADE ===
        lp = self.lowpass_filter(ecg)
        bp = self.highpass_filter(lp)
        der = self.derivative_filter(bp)
        sq = der ** 2
        mwi = self.moving_window_integrator(sq)

        # === PEAK DETECTION WITH ADAPTIVE THRESHOLDS ===
        peak_indices = self._find_peaks(mwi)
        if len(peak_indices) == 0:
            return DetectionResult(
                rpeaks=np.array([], dtype=int),
                bandpassed=bp, derivative=der, squared=sq, integrated=mwi,
                peak_indices=peak_indices,
                heart_rate_bpm=np.array([]),
                rr_intervals_sec=np.array([]),
            )

        # Initialize thresholds from first 2 seconds
        init_end = min(len(mwi), self.learn_n)
        init_peaks = peak_indices[peak_indices < init_end]
        if len(init_peaks) > 0:
            vals_i = mwi[init_peaks]
            vals_f = np.abs(bp[init_peaks])
            spki = 0.25 * np.max(vals_i)
            npki = 0.5 * np.mean(vals_i)
            spkf = 0.25 * np.max(vals_f)
            npkf = 0.5 * np.mean(vals_f)
        else:
            seg_i = mwi[:init_end]
            seg_f = np.abs(bp[:init_end])
            spki = 0.25 * np.max(seg_i) if len(seg_i) else 1e-6
            npki = 0.5 * np.mean(seg_i) if len(seg_i) else 1e-6
            spkf = 0.25 * np.max(seg_f) if len(seg_f) else 1e-6
            npkf = 0.5 * np.mean(seg_f) if len(seg_f) else 1e-6

        alpha = self.cfg.peak_ema_alpha
        tw = self.cfg.threshold_weight

        def thr_i():
            return npki + tw * (spki - npki)

        def thr_f():
            return npkf + tw * (spkf - npkf)

        qrs_inds: List[int] = []
        rr_all: deque = deque(maxlen=8)
        rr_ok: deque = deque(maxlen=8)
        last_qrs = -(10 ** 9)
        last_qrs_slope = None
        search_radius = max(1, int(round(0.150 * self.cfg.fs)))

        for k, pk in enumerate(peak_indices):
            pki = mwi[pk]
            pf_idx = self._search_local_peak(np.abs(bp), pk, search_radius)
            pkf = abs(bp[pf_idx])

            is_refractory = (pk - last_qrs) < self.refractory

            if pki >= thr_i() and pkf >= thr_f() and not is_refractory:
                # T-wave discrimination
                is_t_wave = False
                if qrs_inds and (pk - last_qrs) < self.t_wave_window:
                    curr_slope = np.max(np.abs(der[max(0, pk - 10):min(len(der), pk + 10)]))
                    if last_qrs_slope is not None and curr_slope < 0.5 * last_qrs_slope:
                        is_t_wave = True

                if is_t_wave:
                    npki = alpha * pki + (1 - alpha) * npki
                    npkf = alpha * pkf + (1 - alpha) * npkf
                    continue

                # Accept as QRS
                qrs_inds.append(pf_idx)
                spki = alpha * pki + (1 - alpha) * spki
                spkf = alpha * pkf + (1 - alpha) * spkf

                if len(qrs_inds) >= 2:
                    rr = qrs_inds[-1] - qrs_inds[-2]
                    rr_all.append(rr)
                    if len(rr_ok) == 0:
                        rr_ok.append(rr)
                    else:
                        rr_avg = float(np.mean(rr_ok))
                        if self.cfg.rr_low_factor * rr_avg <= rr <= self.cfg.rr_high_factor * rr_avg:
                            rr_ok.append(rr)

                last_qrs = pk
                last_qrs_slope = np.max(np.abs(der[max(0, pk - 10):min(len(der), pk + 10)]))
            else:
                npki = alpha * pki + (1 - alpha) * npki
                npkf = alpha * pkf + (1 - alpha) * npkf

            # === SEARCHBACK ===
            if qrs_inds and len(rr_all) > 0:
                rr_ref = float(np.mean(rr_ok)) if len(rr_ok) > 0 else float(np.mean(rr_all))
                rr_missed = int(round(self.cfg.searchback_rr_factor * rr_ref))
                next_pk = peak_indices[k + 1] if (k + 1) < len(peak_indices) else len(mwi) - 1

                if next_pk - qrs_inds[-1] > rr_missed:
                    sb_lo = qrs_inds[-1] + self.refractory
                    sb_hi = next_pk
                    if sb_lo < sb_hi:
                        sb_cands = peak_indices[(peak_indices >= sb_lo) & (peak_indices < sb_hi)]
                        if len(sb_cands) > 0:
                            sb_best = sb_cands[np.argmax(mwi[sb_cands])]
                            sb_pki = float(mwi[sb_best])
                            sb_pf = self._search_local_peak(np.abs(bp), int(sb_best), search_radius)
                            sb_pkf = float(abs(bp[sb_pf]))

                            if sb_pki >= 0.5 * thr_i() and sb_pkf >= 0.5 * thr_f():
                                qrs_inds.append(sb_pf)
                                spki = alpha * sb_pki + (1 - alpha) * spki
                                spkf = alpha * sb_pkf + (1 - alpha) * spkf

                                if len(qrs_inds) >= 2:
                                    rr = qrs_inds[-1] - qrs_inds[-2]
                                    rr_all.append(rr)
                                    rr_avg = float(np.mean(rr_ok)) if len(rr_ok) > 0 else rr
                                    if self.cfg.rr_low_factor * rr_avg <= rr <= self.cfg.rr_high_factor * rr_avg:
                                        rr_ok.append(rr)

                                last_qrs = int(sb_best)
                                last_qrs_slope = np.max(
                                    np.abs(der[max(0, int(sb_best) - 10):min(len(der), int(sb_best) + 10)])
                                )

        # Finalize
        rpeaks = np.array(sorted(set(qrs_inds)), dtype=int)

        # Compute RR intervals and heart rate
        if len(rpeaks) >= 2:
            rr_samples = np.diff(rpeaks)
            rr_sec = rr_samples / self.cfg.fs
            hr_bpm = 60.0 / rr_sec
        else:
            rr_sec = np.array([])
            hr_bpm = np.array([])

        return DetectionResult(
            rpeaks=rpeaks,
            bandpassed=bp,
            derivative=der,
            squared=sq,
            integrated=mwi,
            peak_indices=peak_indices,
            heart_rate_bpm=hr_bpm,
            rr_intervals_sec=rr_sec,
            debug={
                "spki": spki, "npki": npki, "spkf": spkf, "npkf": npkf,
                "threshold_i": thr_i(), "threshold_f": thr_f(),
                "integration_window": self.int_win,
                "refractory_samples": self.refractory,
            },
        )


class AdaptivePanTompkins:
    """
    Multi-rate adaptive Pan-Tompkins.

    Auto-resamples input to 200Hz for the exact integer filter cascade,
    runs detection, and maps peaks back to the original sampling rate.
    """

    def __init__(self):
        self.detector = PanTompkinsDetector(PanTompkinsConfig(fs=200.0))

    def detect(self, ecg: np.ndarray, fs: int) -> DetectionResult:
        """
        Detect R-peaks in ECG signal at any sampling rate.

        Args:
            ecg: 1D ECG signal
            fs: Sampling rate of the input signal

        Returns:
            DetectionResult with peaks mapped back to original rate
        """
        from scipy.signal import resample_poly
        from math import gcd

        ecg = np.asarray(ecg, dtype=np.float64).squeeze()

        if fs == 200:
            return self.detector.detect(ecg)

        # Resample to 200Hz
        g = gcd(200, fs)
        ecg_200 = resample_poly(ecg, 200 // g, fs // g)

        # Detect at 200Hz
        result = self.detector.detect(ecg_200)

        # Map peaks back to original rate
        if len(result.rpeaks) > 0:
            # Convert indices and snap to local max in original signal
            mapped_peaks = np.round(result.rpeaks * fs / 200.0).astype(int)
            mapped_peaks = np.clip(mapped_peaks, 0, len(ecg) - 1)

            # Snap to local maximum in original signal within ±10 samples
            snap_radius = max(1, int(round(0.01 * fs)))
            for i, pk in enumerate(mapped_peaks):
                lo = max(0, pk - snap_radius)
                hi = min(len(ecg), pk + snap_radius + 1)
                mapped_peaks[i] = lo + np.argmax(ecg[lo:hi])

            result.rpeaks = mapped_peaks

            # Recompute RR and HR at original rate
            if len(mapped_peaks) >= 2:
                rr_samples = np.diff(mapped_peaks)
                result.rr_intervals_sec = rr_samples / fs
                result.heart_rate_bpm = 60.0 / result.rr_intervals_sec

        return result
