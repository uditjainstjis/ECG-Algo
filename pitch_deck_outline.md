# HeartEngine Pitch Deck Outline
*(Feed this directly into NotebookLM along with project_report.md to generate your presentation slides)*

## Slide 1: Title Slide
* **Title:** HeartEngine
* **Subtitle:** Real-Time, 4-Stage Hybrid ECG Telemetry & Arrhythmia Detection
* **Visual Idea:** A sleek, glassmorphism UI showing a live ECG wave transitioning from raw noise into a clean, classified signal.

## Slide 2: The Problem
* **Legacy Systems are Brittle:** Standard hospital ECGs are bulky, offline, and rely on 1980s math.
* **Wearables are Noisy:** Modern wearables (Apple Watch, Arduino) suffer from extreme motion artifacts, making raw deep learning fail spectacularly due to "garbage-in, garbage-out."
* **Hardware/Software Disconnect:** Web-based dashboards routinely crash or lock up when trying to read continuous, high-speed serial USB telemetry streams.

## Slide 3: The Solution (HeartEngine)
* **A 4-Stage Hybrid Approach:** We don't just rely on Deep Learning. We fuse classical Digital Signal Processing (DSP) with modern Foundation Models.
* **Hardware-Agnostic:** Capable of reading legacy clinical Holter data (WFDB) and real-time proprietary streaming formats (10-byte `.aecg`).
* **Zero-Crash Architecture:** A decoupled terminal proxy safely handles the hardware, while the web dashboard reads memory-mapped buffers for a butter-smooth 60FPS UI.

## Slide 4: Stage 1 & 2 - DSP and Feature Extraction
* **Adaptive Pan-Tompkins DSP:** An exact 200Hz integer-based filter cascade. It cleans the noise and acts as our "source of truth."
* **HRV Engine:** We extract 28 complex Heart Rate Variability (HRV) metrics instantly, capturing the non-linear chaos of the heart.
* **Accuracy:** R-Peak detection operates at **99.96% Sensitivity**.

## Slide 5: Stage 3 - The AI (XGBoost & Isolation Forest)
* **XGBoost Classifier:** Uses the 28 HRV features to detect Atrial Fibrillation with a flawless **1.00 F1-Score** on clinical validation sets.
* **Unsupervised Anomaly Detection (The Wow Factor):** We deployed an Isolation Forest trained *only on healthy hearts*. It doesn't look for diseases; it looks for deviations from health. It flags PVCs, PACs, and noise dynamically—without needing a single labeled disease example during training.

## Slide 6: Stage 4 - Audio Transfer Learning (The Innovation)
* **Wav2Vec2 Foundation Model:** An ECG is just a low-frequency sound wave. We adapted Facebook's 95-million parameter audio speech model using Rank-16 LoRA (Parameter-Efficient Fine-Tuning).
* **Cross-Domain Intelligence:** By treating the heart like audio, we achieved complex morphological feature extraction at a fraction of the computational cost of standard CNNs.

## Slide 7: The Smart Ensemble (SQI)
* **How it thinks:** The engine dynamically calculates a Signal Quality Index (SQI) in real-time.
* **High Noise (Patient is running):** The system disables Deep Learning and trusts the mathematical Pan-Tompkins DSP.
* **Clean Signal (Patient is resting):** The system unleashes the Foundation Models for deep classification.
* **End-to-End Latency:** < 120ms.

## Slide 8: The Clinical Output & Live Demo
* **From Numbers to English:** The system synthesizes all metrics into a clinical narrative report ready for LLM summarization and doctor review.
* **Live Demo Transition:** (At this point, you unplug and plug the Arduino to show the robust auto-reconnect and live streaming capabilities without crashing the Streamlit dashboard).

## Slide 9: Conclusion & Future Roadmap
* **What we achieved today:** A fully functioning, hardware-coupled, foundation-model-driven diagnostic engine.
* **Where we go tomorrow:** Deployment to edge devices (Raspberry Pi/Coral TPU), expanding the unsupervised anomaly detector to federated learning networks across hospitals.
