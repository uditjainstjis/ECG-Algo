"""
ECGFounder ONNX Inference Wrapper
====================================
Zero-GPU foundation model inference via ECGFounder ONNX export.
Supports single-lead input via channel repetition.
Returns 71-class diagnostic probabilities including AF.
"""

import numpy as np
import logging
import os
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# ECGFounder diagnostic classes (subset relevant to AF)
ECGFOUNDER_CLASSES = {
    0: "Normal_Sinus_Rhythm",
    1: "Atrial_Fibrillation",
    2: "Atrial_Flutter",
    3: "First_degree_AV_block",
    4: "Left_bundle_branch_block",
    5: "Right_bundle_branch_block",
    6: "Premature_atrial_contraction",
    7: "Premature_ventricular_contraction",
    8: "ST_depression",
    9: "ST_elevation",
    # ... remaining classes from the 71 full list
}

AF_CLASS_IDX = 1  # Atrial Fibrillation index


class ECGFounderONNX:
    """
    ECGFounder inference via ONNX Runtime.
    Runs on CPU — no GPU required.
    """

    def __init__(self, model_path: Optional[str] = None):
        try:
            import onnxruntime as ort
            self.ort = ort
        except ImportError:
            raise ImportError("onnxruntime not installed. pip install onnxruntime")

        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None

    def download_model(self, cache_dir: str = "models") -> str:
        """Download ECGFounder ONNX model from HuggingFace."""
        from huggingface_hub import hf_hub_download

        os.makedirs(cache_dir, exist_ok=True)
        path = hf_hub_download(
            repo_id="ghirani33/ecgfounder-onnx",
            filename="ecg_founder_all71.onnx",
            cache_dir=cache_dir,
        )
        self.model_path = path
        logger.info(f"Downloaded ECGFounder ONNX to: {path}")
        return path

    def load(self):
        """Load the ONNX session."""
        if self.model_path is None:
            raise RuntimeError("No model path. Call download_model() first.")

        self.session = self.ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        logger.info(f"ECGFounder ONNX loaded. Input shape: {input_shape}")

    def preprocess_single_lead(
        self,
        ecg: np.ndarray,
        fs: int,
        target_length: int = 5000,
        target_fs: int = 500,
    ) -> np.ndarray:
        """
        Prepare single-lead ECG for ECGFounder (expects 12-lead × 5000).

        Strategy: Resample to 500Hz, pad/truncate to 5000 samples,
        repeat single lead to 12 channels.
        """
        from scipy.signal import resample_poly
        from math import gcd

        ecg = np.asarray(ecg, dtype=np.float32).squeeze()

        # Resample to 500Hz
        if fs != target_fs:
            g = gcd(target_fs, fs)
            ecg = resample_poly(ecg, target_fs // g, fs // g).astype(np.float32)

        # Pad or truncate to target_length
        if len(ecg) < target_length:
            ecg = np.pad(ecg, (0, target_length - len(ecg)), mode="edge")
        elif len(ecg) > target_length:
            ecg = ecg[:target_length]

        # Normalize
        mu, std = np.mean(ecg), np.std(ecg)
        if std > 1e-6:
            ecg = (ecg - mu) / std

        # Repeat to 12 leads: (1, 12, 5000)
        ecg_12 = np.tile(ecg, (12, 1))
        return ecg_12[np.newaxis, ...].astype(np.float32)

    def predict(
        self,
        ecg: np.ndarray,
        fs: int,
    ) -> Dict[str, float]:
        """
        Run inference on single-lead ECG.

        Args:
            ecg: 1D ECG signal
            fs: Sampling rate

        Returns:
            Dict with class probabilities, AF probability, and predicted class
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Preprocess
        input_tensor = self.preprocess_single_lead(ecg, fs)

        # Run inference
        logits = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        probs = _softmax(logits[0])

        # Extract key results
        af_prob = float(probs[AF_CLASS_IDX]) if AF_CLASS_IDX < len(probs) else 0.0
        pred_idx = int(np.argmax(probs))
        pred_class = ECGFOUNDER_CLASSES.get(pred_idx, f"Class_{pred_idx}")

        return {
            "af_probability": af_prob,
            "predicted_class": pred_class,
            "predicted_class_idx": pred_idx,
            "confidence": float(probs[pred_idx]),
            "all_probabilities": probs.tolist(),
        }

    def predict_window_sequence(
        self,
        signal: np.ndarray,
        fs: int,
        window_sec: float = 10.0,
        stride_sec: float = 5.0,
    ) -> list:
        """Run inference on sliding windows across a recording."""
        window_samples = int(window_sec * fs)
        stride_samples = int(stride_sec * fs)
        results = []

        for start in range(0, len(signal) - window_samples + 1, stride_samples):
            end = start + window_samples
            window = signal[start:end]
            result = self.predict(window, fs)
            result["start_sec"] = start / fs
            result["end_sec"] = end / fs
            results.append(result)

        return results


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
