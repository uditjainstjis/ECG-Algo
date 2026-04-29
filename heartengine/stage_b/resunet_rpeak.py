"""
1D Residual U-Net for R-Peak Segmentation
============================================
Treats R-peak detection as 1D dense segmentation.
5-level encoder-decoder with residual bottleneck blocks,
squeeze-excitation attention, attention gates on skip connections,
and per-sample sigmoid probability map output.

Training target: Gaussian blobs centered on annotated R-peaks.
Loss: Focal + Dice composite.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ==================== BUILDING BLOCKS ====================

class SqueezeExcitation1D(nn.Module):
    """Channel attention via global average pool → FC → sigmoid gating."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class ResidualBottleneck1D(nn.Module):
    """Residual bottleneck: 1×1 reduce → 3×1 conv → 1×1 restore + SE."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, ratio: int = 4):
        super().__init__()
        mid = max(out_ch // ratio, 16)
        self.conv1 = nn.Conv1d(in_ch, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid)
        self.conv2 = nn.Conv1d(mid, mid, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(mid)
        self.conv3 = nn.Conv1d(mid, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_ch)
        self.se = SqueezeExcitation1D(out_ch)
        self.proj = (
            nn.Identity() if (in_ch == out_ch and stride == 1)
            else nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        )

    def forward(self, x):
        identity = self.proj(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        return F.relu(out + identity, inplace=True)


class AttentionGate1D(nn.Module):
    """Attention gate for skip connections — suppresses non-QRS features."""
    def __init__(self, x_ch: int, g_ch: int, inter_ch: int):
        super().__init__()
        self.wx = nn.Conv1d(x_ch, inter_ch, 1, bias=False)
        self.wg = nn.Conv1d(g_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_ch, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, g):
        return x * self.psi(self.wx(x) + self.wg(g))


class DecoderBlock1D(nn.Module):
    """Upsample + attention-gated skip + double conv refine."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attn = AttentionGate1D(skip_ch, out_ch, inter_ch=max(out_ch // 2, 16))
        self.conv = nn.Sequential(
            nn.Conv1d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from non-power-of-2 lengths
        if x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
        skip = self.attn(skip, x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ==================== MAIN MODEL ====================

class ResUNet1D(nn.Module):
    """
    1D Residual U-Net for R-peak segmentation.

    Architecture: 5-level encoder (64→128→256→512→1024) with residual
    bottleneck blocks and SE attention, decoder with attention-gated
    skip connections, 1×1 sigmoid output head.

    Input:  (B, 1, T) — single-lead ECG window
    Output: (B, 1, T) — per-sample R-peak probability
    """

    def __init__(self, in_ch: int = 1, base: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
        )

        # Encoder: 5 levels
        self.e1 = ResidualBottleneck1D(base, base)
        self.e2 = ResidualBottleneck1D(base, base * 2, stride=2)
        self.e3 = ResidualBottleneck1D(base * 2, base * 4, stride=2)
        self.e4 = ResidualBottleneck1D(base * 4, base * 8, stride=2)
        self.e5 = ResidualBottleneck1D(base * 8, base * 16, stride=2)

        # Decoder: 4 levels
        self.d4 = DecoderBlock1D(base * 16, base * 8, base * 8)
        self.d3 = DecoderBlock1D(base * 8, base * 4, base * 4)
        self.d2 = DecoderBlock1D(base * 4, base * 2, base * 2)
        self.d1 = DecoderBlock1D(base * 2, base, base)

        # Output head
        self.head = nn.Conv1d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        s1 = self.e1(x0)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        bottleneck = self.e5(s4)

        x = self.d4(bottleneck, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)

        return torch.sigmoid(self.head(x))


# ==================== TARGET GENERATION ====================

def generate_gaussian_target(
    length: int,
    rpeak_indices: np.ndarray,
    sigma: int = 5,
) -> np.ndarray:
    """
    Generate Gaussian soft target for R-peak segmentation.

    Args:
        length: Length of the output target array
        rpeak_indices: Indices of R-peak locations
        sigma: Gaussian standard deviation (samples)

    Returns:
        1D numpy array with Gaussian blobs at R-peak locations, values in [0, 1]
    """
    target = np.zeros(length, dtype=np.float32)
    t = np.arange(length)

    for r in rpeak_indices:
        if 0 <= r < length:
            gaussian = np.exp(-0.5 * ((t - r) / sigma) ** 2)
            target = np.maximum(target, gaussian)

    return np.clip(target, 0.0, 1.0)


# ==================== PEAK DECODER ====================

def decode_peaks(
    prob_map: np.ndarray,
    fs: int,
    threshold: float = 0.3,
    refractory_ms: float = 200.0,
    snap_signal: Optional[np.ndarray] = None,
    snap_radius_ms: float = 40.0,
) -> np.ndarray:
    """
    Post-process probability map into R-peak indices.

    Pipeline: threshold → local maxima → refractory suppression → snap to raw ECG max
    """
    # Threshold
    above = prob_map > threshold

    # Find local maxima in thresholded regions
    if len(prob_map) < 3:
        return np.array([], dtype=int)

    local_max = np.zeros(len(prob_map), dtype=bool)
    for i in range(1, len(prob_map) - 1):
        if above[i] and prob_map[i] > prob_map[i-1] and prob_map[i] >= prob_map[i+1]:
            local_max[i] = True

    peaks = np.where(local_max)[0]
    if len(peaks) == 0:
        return np.array([], dtype=int)

    # Sort by confidence (descending)
    order = np.argsort(-prob_map[peaks])
    peaks = peaks[order]

    # Refractory suppression (NMS)
    refractory_samples = int(refractory_ms * fs / 1000.0)
    accepted = []
    for p in peaks:
        if all(abs(p - a) >= refractory_samples for a in accepted):
            accepted.append(p)
    accepted = sorted(accepted)

    # Snap to local maximum of raw ECG
    if snap_signal is not None:
        snap_radius = int(snap_radius_ms * fs / 1000.0)
        for i, p in enumerate(accepted):
            lo = max(0, p - snap_radius)
            hi = min(len(snap_signal), p + snap_radius + 1)
            accepted[i] = lo + int(np.argmax(snap_signal[lo:hi]))

    return np.array(accepted, dtype=int)
