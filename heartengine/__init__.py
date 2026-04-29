"""
HeartEngine: 4-Stage Hybrid ECG Analysis System
================================================

A production-grade system that layers Classical DSP → Deep Learning →
Foundation Model → LLM Narrative for heart rate detection and atrial
fibrillation classification from single-lead ECG.

Stages:
    A: Classical DSP (Pan-Tompkins) + ML (XGBoost on HRV features)
    B: Task-Specific Deep Learning (ResU-Net R-peak + CNN-Transformer AFib)
    C: ECG Foundation Model (ECGFounder ONNX + ECG-FM LoRA)
    D: LLM Clinical Narrative Generation
"""

__version__ = "1.0.0"
__author__ = "HeartEngine Research Team"
