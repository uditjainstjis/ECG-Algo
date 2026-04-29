import struct
import torch
import numpy as np
from transformers import Wav2Vec2Model
import torch.nn as nn

class Wav2Vec2ForECGClassification(nn.Module):
    """Same architecture used in training for inference."""
    def __init__(self, model_name="facebook/wav2vec2-base", num_classes=2):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, input_values):
        outputs = self.backbone(input_values)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        return self.classifier(pooled)


def read_hackathon_binary(file_path):
    """
    Reads the custom 10-byte hackathon format.
    Byte 0-1: int16 (ECG Value)
    Byte 2-9: int64 little-endian (Timestamp in ms)
    """
    ecg_values = []
    timestamps = []
    
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(10)
            if not chunk or len(chunk) < 10:
                break
            
            # <h means little-endian 16-bit signed int
            # <q means little-endian 64-bit signed int
            val, ts = struct.unpack('<hq', chunk)
            ecg_values.append(val)
            timestamps.append(ts)
            
    return np.array(ecg_values, dtype=np.float32), np.array(timestamps)


def run_hackathon_inference(binary_file_path, adapter_dir="gpu_training (2)/lora_ecg_adapter"):
    """
    Reads a hackathon binary file and runs the LoRA adapter to detect AFib.
    """
    from peft import PeftModel
    
    print(f"Reading hackathon binary file: {binary_file_path}")
    signal, timestamps = read_hackathon_binary(binary_file_path)
    
    # Normalize signal
    mu, std = np.mean(signal), np.std(signal)
    if std > 1e-6:
        signal = (signal - mu) / std
        
    # Resample to 16kHz (wav2vec2 requirement)
    # We infer original sampling rate from timestamps
    if len(timestamps) > 1:
        dt_ms = timestamps[1] - timestamps[0]
        original_fs = 1000.0 / dt_ms
    else:
        original_fs = 250.0 # Default fallback
        
    print(f"Detected original sampling rate: {original_fs} Hz")
    
    from scipy.signal import resample
    duration_sec = len(signal) / original_fs
    target_len = int(duration_sec * 16000)
    
    print("Resampling to 16kHz for Wav2Vec2...")
    signal_16k = resample(signal, target_len).astype(np.float32)
    
    # Load Model
    print("Loading Base Model and LoRA Adapter...")
    base_model = Wav2Vec2ForECGClassification("facebook/wav2vec2-base")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    
    # Inference
    input_tensor = torch.tensor(signal_16k).unsqueeze(0) # (1, T)
    
    print("Running Inference...")
    with torch.no_grad():
        logits = model(input_tensor)
        prediction = logits.argmax(dim=1).item()
        
    result = "AFIB (Atrial Fibrillation)" if prediction == 1 else "Normal Rhythm"
    print(f"\n[RESULT] The model diagnosed this file as: {result}")
    return prediction

if __name__ == "__main__":
    print("Hackathon Binary Format Parser Ready.")
    # To test during hackathon:
    # run_hackathon_inference("test_subject.bin")
