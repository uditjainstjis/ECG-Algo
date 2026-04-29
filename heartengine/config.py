"""
HeartEngine Unified Configuration
==================================
Central configuration for all stages, data paths, and hyperparameters.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PathConfig:
    """File system paths."""
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = field(init=False)
    MODELS_DIR: str = field(init=False)
    RESULTS_DIR: str = field(init=False)

    def __post_init__(self):
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.MODELS_DIR = os.path.join(self.BASE_DIR, "models")
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, "results")
        for d in [self.DATA_DIR, self.MODELS_DIR, self.RESULTS_DIR]:
            os.makedirs(d, exist_ok=True)


@dataclass
class SignalConfig:
    """ECG signal processing parameters."""
    TARGET_FS: int = 250            # Target sampling rate (Hz)
    PAN_TOMPKINS_FS: int = 200      # Pan-Tompkins native rate
    RPEAK_WINDOW_SEC: float = 5.0   # Window size for R-peak detection (seconds)
    RPEAK_OVERLAP: float = 0.5      # Window overlap ratio
    RHYTHM_WINDOW_SEC: float = 30.0 # Window size for rhythm classification
    RHYTHM_STRIDE_SEC: float = 10.0 # Stride for rhythm scanning

    # Preprocessing
    BANDPASS_LOW: float = 0.5       # Bandpass lower cutoff (Hz)
    BANDPASS_HIGH: float = 40.0     # Bandpass upper cutoff (Hz)
    QRS_BANDPASS_LOW: float = 5.0   # QRS-enhancement bandpass lower
    QRS_BANDPASS_HIGH: float = 15.0 # QRS-enhancement bandpass upper
    NOTCH_FREQ: float = 50.0       # Powerline frequency (50Hz EU / 60Hz US)


@dataclass
class StageAConfig:
    """Classical DSP + ML configuration."""
    # SQI thresholds
    SQI_THRESHOLD: float = 0.4
    SQI_KURTOSIS_WEIGHT: float = 0.3
    SQI_SPECTRAL_WEIGHT: float = 0.3
    SQI_POWER_WEIGHT: float = 0.2
    SQI_BEAT_AGREEMENT_WEIGHT: float = 0.2

    # XGBoost
    XGB_MAX_DEPTH: int = 6
    XGB_N_ESTIMATORS: int = 200
    XGB_LEARNING_RATE: float = 0.1
    XGB_SUBSAMPLE: float = 0.8

    # HRV window
    HRV_WINDOW_BEATS: int = 100
    HRV_STRIDE_BEATS: int = 50


@dataclass
class StageBConfig:
    """Deep Learning configuration."""
    # ResU-Net
    RESUNET_BASE_FILTERS: int = 64
    RESUNET_LEVELS: int = 5
    GAUSSIAN_SIGMA_250HZ: int = 5
    GAUSSIAN_SIGMA_360HZ: int = 7
    FOCAL_GAMMA: float = 2.0
    FOCAL_ALPHA: float = 0.75
    DICE_WEIGHT: float = 0.3
    FOCAL_WEIGHT: float = 0.7
    PEAK_THRESHOLD: float = 0.3
    REFRACTORY_MS: float = 200.0
    SNAP_RADIUS_MS: float = 40.0

    # CNN-Transformer AFib
    CNN_D_MODEL: int = 128
    CNN_NHEAD: int = 4
    CNN_NUM_LAYERS: int = 3
    CNN_FFN_DIM: int = 512

    # Training
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-3
    MIN_LR: float = 1e-5
    EPOCHS: int = 200
    RPEAK_TOLERANCE_MS: float = 75.0


@dataclass
class StageCConfig:
    """Foundation Model configuration."""
    ECGFOUNDER_ONNX_REPO: str = "ghirani33/ecgfounder-onnx"
    ECGFOUNDER_ONNX_FILE: str = "ecg_founder_all71.onnx"
    ECGFM_REPO: str = "wanglab/ecg-fm"
    ECGFM_CHECKPOINT: str = "mimic_iv_ecg_physionet_pretrained.pt"

    # LoRA
    LORA_RANK: int = 8
    LORA_ALPHA: int = 16
    LORA_DROPOUT: float = 0.1

    # Input
    ECG_INPUT_LENGTH: int = 5000  # 10 seconds at 500Hz
    NUM_LEADS_EXPECTED: int = 12


@dataclass
class HeartEngineConfig:
    """Master configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    stage_a: StageAConfig = field(default_factory=StageAConfig)
    stage_b: StageBConfig = field(default_factory=StageBConfig)
    stage_c: StageCConfig = field(default_factory=StageCConfig)
    device: str = field(init=False)
    seed: int = 42

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"


# Global config singleton
CONFIG = HeartEngineConfig()
