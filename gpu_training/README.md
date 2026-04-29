# RTX 5090 GPU Training Package

## 🚀 Quick Start

```bash
# 1. Copy this folder to the GPU machine
scp -r gpu_training/ user@gpu-box:~/heartengine/

# 2. SSH into the machine
ssh user@gpu-box

# 3. Install dependencies
pip install torch numpy scipy wfdb tqdm

# 4. Run training (auto-downloads data from PhysioNet)
cd ~/heartengine
python gpu_training.py
```

## What Gets Trained

### Model 1: ResU-Net R-Peak Segmentation
- **Architecture**: 5-level encoder-decoder, 6.7M params
- **Dataset**: MIT-BIH Arrhythmia (48 records, 360Hz)
- **Training**: 150 epochs, Focal+Dice loss, cosine annealing
- **Expected time**: ~60-90 minutes on RTX 5090
- **Output**: `resunet_rpeak_best.pt`

### Model 2: CNN-Transformer AFib Classifier
- **Architecture**: CNN stem + 3-layer Transformer, 713K params
- **Dataset**: MIT-BIH AFDB (25 records, 10-hour AF recordings)
- **Training**: 100 epochs, cross-entropy with class weights
- **Expected time**: ~30-45 minutes on RTX 5090
- **Output**: `cnn_transformer_afib_best.pt`

## Expected Output

```
Device: cuda
GPU: NVIDIA GeForce RTX 5090
VRAM: 32.0 GB

Downloading MIT-BIH Arrhythmia Database...
Downloading MIT-BIH AF Database...

TRAINING ResU-Net R-Peak Segmentation
ResU-Net params: 6,723,869
...
Epoch 150/150 — Train: 0.01234, Val: 0.01567
✓ Saved best model (val_loss=0.01234)

TRAINING CNN-Transformer AFib Classifier
CNN-Transformer params: 713,059
...
Epoch 100/100 — Train: loss=0.0345 acc=0.9876 | Val: loss=0.0567 acc=0.9654
✓ Saved best model (val_acc=0.9654)

ALL TRAINING COMPLETE in 120.5 minutes
```

## After Training

Copy the model files back to your Mac:

```bash
scp user@gpu-box:~/heartengine/resunet_rpeak_best.pt ~/Desktop/Heart/models/
scp user@gpu-box:~/heartengine/cnn_transformer_afib_best.pt ~/Desktop/Heart/models/
```

Then on your Mac, the pipeline will automatically load these trained models.

## Memory Usage

- ResU-Net training: ~8-10 GB VRAM peak
- CNN-Transformer training: ~4-6 GB VRAM peak
- Well within RTX 5090's 32GB
