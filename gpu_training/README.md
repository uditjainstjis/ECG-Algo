# RTX 5090 GPU Training Package

## 📦 What's in this folder

| Script | What it trains | Time | Output |
|--------|---------------|------|--------|
| `gpu_training.py` | ResU-Net R-peak + CNN-Transformer AFib | ~2 hours | `*_best.pt` files |
| `train_lora_adapter.py` | LoRA adapter on wav2vec2-base for AFib | ~30-60 min | `lora_ecg_adapter/` dir |

## 🚀 Instructions

### Step 1: Install dependencies
```bash
pip install torch transformers peft numpy scipy wfdb tqdm
```

### Step 2: Run CNN training (do this first)
```bash
python gpu_training.py
```

### Step 3: Run LoRA adapter training (after step 2 finishes)
```bash
python train_lora_adapter.py
```

### Step 4: Copy results back to Mac
```bash
scp resunet_rpeak_best.pt user@mac:~/Desktop/Heart/models/
scp cnn_transformer_afib_best.pt user@mac:~/Desktop/Heart/models/
scp -r lora_ecg_adapter/ user@mac:~/Desktop/Heart/models/
```

## 🧠 Why wav2vec2 + LoRA?

**wav2vec2-base** is a 95M-param Transformer pre-trained on 960 hours of audio waveforms. ECG-FM (the leading ECG foundation model, 2024) is literally built on the wav2vec2 architecture — proving this is the right backbone for cardiac signals.

**LoRA** injects tiny rank-16 matrices into the Q/K/V attention projections, training only 0.2% of parameters. The adapter file is ~1-2 MB vs the 360MB base model.

This is the same paradigm as GPT + LoRA adapters — freeze the foundation, specialize with a tiny adapter.
