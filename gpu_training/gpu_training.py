#!/usr/bin/env python3
"""
=========================================================================
  RTX 5090 GPU TRAINING PACKAGE — ResU-Net R-Peak + CNN-Transformer AFib
=========================================================================

SELF-CONTAINED training script for the RTX 5090 (32GB VRAM).
Trains BOTH models in sequence:
  1. ResU-Net 1D R-peak segmentation on MIT-BIH (360Hz, ~48 records)
  2. CNN-Transformer AFib classifier on MIT-BIH AFDB (250Hz, ~25 records)

INSTRUCTIONS FOR GPU BOX:
  1. Copy this entire file to the GPU machine
  2. pip install torch numpy scipy wfdb tqdm
  3. python gpu_training.py
  4. Copy back the saved models: resunet_rpeak_best.pt, cnn_transformer_afib_best.pt

Expected time: ~2-3 hours total on RTX 5090
Expected VRAM: ~8-12 GB peak
"""

import os
import sys
import time
import math
import logging
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

import wfdb
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GPU-Training")

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
DATA_DIR = "./data"

# ResU-Net config
RESUNET_EPOCHS = 150
RESUNET_BATCH = 32
RESUNET_LR = 1e-3
RESUNET_MIN_LR = 1e-5
RESUNET_WINDOW_SEC = 5.0
RESUNET_OVERLAP = 0.5
GAUSSIAN_SIGMA = 7  # for 360Hz

# CNN-Transformer config
AFIB_EPOCHS = 100
AFIB_BATCH = 16
AFIB_LR = 5e-4
AFIB_MIN_LR = 1e-5
AFIB_WINDOW_SEC = 30.0

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# DATA DOWNLOAD & LOADING
# ============================================================

def download_datasets():
    """Download MIT-BIH and AFDB from PhysioNet."""
    os.makedirs(DATA_DIR, exist_ok=True)

    mitdb_dir = os.path.join(DATA_DIR, "mitdb")
    afdb_dir = os.path.join(DATA_DIR, "afdb")

    if not os.path.exists(mitdb_dir) or len([f for f in os.listdir(mitdb_dir) if f.endswith(".dat")]) < 40:
        logger.info("Downloading MIT-BIH Arrhythmia Database...")
        wfdb.dl_database("mitdb", dl_dir=mitdb_dir)

    if not os.path.exists(afdb_dir) or len([f for f in os.listdir(afdb_dir) if f.endswith(".dat")]) < 20:
        logger.info("Downloading MIT-BIH AF Database...")
        wfdb.dl_database("afdb", dl_dir=afdb_dir)

    return mitdb_dir, afdb_dir


def load_record_single_lead(record_path, target_fs=None):
    """Load a single-lead ECG from WFDB record."""
    record = wfdb.rdrecord(record_path)
    fs = record.fs
    signal = record.p_signal[:, 0].astype(np.float64)

    # Load R-peak annotations
    r_peaks = np.array([], dtype=int)
    rhythm_labels = []
    try:
        ann = wfdb.rdann(record_path, "atr")
        beat_types = set("NLRBAaJSVrFejn/fQ?")
        beat_mask = np.array([s in beat_types for s in ann.symbol])
        r_peaks = ann.sample[beat_mask]

        for i, (samp, sym, aux) in enumerate(zip(ann.sample, ann.symbol, ann.aux_note)):
            if sym == "+" and aux.strip():
                end = len(signal)
                for j in range(i + 1, len(ann.sample)):
                    if ann.symbol[j] == "+":
                        end = ann.sample[j]
                        break
                rhythm_labels.append((samp, end, aux.strip()))
    except:
        pass

    # Resample if needed
    if target_fs and target_fs != fs:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(target_fs, fs)
        orig_fs = fs
        signal = resample_poly(signal, target_fs // g, fs // g)
        if len(r_peaks) > 0:
            r_peaks = np.round(r_peaks * target_fs / orig_fs).astype(int)
            r_peaks = np.clip(r_peaks, 0, len(signal) - 1)
        rhythm_labels = [(int(round(s*target_fs/orig_fs)), int(round(e*target_fs/orig_fs)), l)
                         for s, e, l in rhythm_labels]
        fs = target_fs

    return signal, fs, r_peaks, rhythm_labels


# ============================================================
# MODEL ARCHITECTURES (self-contained)
# ============================================================

class SqueezeExcitation1D(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        mid = max(ch // r, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Conv1d(ch, mid, 1), nn.ReLU(True),
            nn.Conv1d(mid, ch, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        mid = max(out_ch // 4, 16)
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, mid, 1, bias=False), nn.BatchNorm1d(mid), nn.ReLU(True),
            nn.Conv1d(mid, mid, 3, stride=stride, padding=1, bias=False), nn.BatchNorm1d(mid), nn.ReLU(True),
            nn.Conv1d(mid, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch))
        self.se = SqueezeExcitation1D(out_ch)
        self.proj = (nn.Identity() if (in_ch == out_ch and stride == 1)
                     else nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm1d(out_ch)))

    def forward(self, x):
        return F.relu(self.se(self.block(x)) + self.proj(x), inplace=True)


class AttnGate(nn.Module):
    def __init__(self, x_ch, g_ch, inter):
        super().__init__()
        self.wx = nn.Conv1d(x_ch, inter, 1, bias=False)
        self.wg = nn.Conv1d(g_ch, inter, 1, bias=False)
        self.psi = nn.Sequential(nn.ReLU(True), nn.Conv1d(inter, 1, 1), nn.Sigmoid())

    def forward(self, x, g):
        return x * self.psi(self.wx(x) + self.wg(g))


class DecBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, 2, stride=2)
        self.attn = AttnGate(skip_ch, out_ch, max(out_ch // 2, 16))
        self.conv = nn.Sequential(
            nn.Conv1d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm1d(out_ch), nn.ReLU(True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm1d(out_ch), nn.ReLU(True))

    def forward(self, x, skip):
        x = self.up(x)
        if x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
        skip = self.attn(skip, x)
        return self.conv(torch.cat([x, skip], 1))


class ResUNet1D(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, base, 7, padding=3, bias=False), nn.BatchNorm1d(base), nn.ReLU(True))
        self.e1 = ResBlock(base, base)
        self.e2 = ResBlock(base, base*2, stride=2)
        self.e3 = ResBlock(base*2, base*4, stride=2)
        self.e4 = ResBlock(base*4, base*8, stride=2)
        self.e5 = ResBlock(base*8, base*16, stride=2)
        self.d4 = DecBlock(base*16, base*8, base*8)
        self.d3 = DecBlock(base*8, base*4, base*4)
        self.d2 = DecBlock(base*4, base*2, base*2)
        self.d1 = DecBlock(base*2, base, base)
        self.head = nn.Conv1d(base, 1, 1)

    def forward(self, x):
        x0 = self.stem(x); s1 = self.e1(x0); s2 = self.e2(s1)
        s3 = self.e3(s2); s4 = self.e4(s3); b = self.e5(s4)
        x = self.d4(b, s4); x = self.d3(x, s3); x = self.d2(x, s2); x = self.d1(x, s1)
        return torch.sigmoid(self.head(x))


class CNNTransformerAFib(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3, num_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, d_model, 3, padding=1, bias=False), nn.BatchNorm1d(d_model), nn.GELU())
        pe = torch.zeros(2000, d_model)
        pos = torch.arange(2000).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        enc = nn.TransformerEncoderLayer(d_model, nhead, 4*d_model, 0.1, batch_first=True, activation="gelu")
        self.tx = nn.TransformerEncoder(enc, num_layers)
        self.pool_attn = nn.Sequential(nn.Linear(d_model, d_model//2), nn.Tanh(), nn.Linear(d_model//2, 1))
        self.fc = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.1),
                                nn.Linear(d_model, d_model//2), nn.GELU(), nn.Dropout(0.1),
                                nn.Linear(d_model//2, num_classes))

    def forward(self, x):
        x = self.cnn(x).transpose(1, 2)
        x = x + self.pe[:, :x.size(1)]
        x = self.tx(x)
        w = torch.softmax(self.pool_attn(x), dim=1)
        z = (x * w).sum(dim=1)
        return self.fc(z)


# ============================================================
# LOSSES
# ============================================================

class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, fw=0.7, dw=0.3):
        super().__init__()
        self.gamma, self.alpha, self.fw, self.dw = gamma, alpha, fw, dw

    def forward(self, pred, target):
        # Focal
        pred_c = pred.clamp(1e-7, 1-1e-7)
        bce = -(target*torch.log(pred_c) + (1-target)*torch.log(1-pred_c))
        pt = target*pred_c + (1-target)*(1-pred_c)
        focal = ((1-pt)**self.gamma * (target*self.alpha + (1-target)*(1-self.alpha)) * bce).mean()
        # Dice
        p, t = pred.view(-1), target.view(-1)
        dice = 1 - (2*(p*t).sum()+1) / (p.sum()+t.sum()+1)
        return self.fw*focal + self.dw*dice


# ============================================================
# DATASETS
# ============================================================

class RPeakSegDataset(Dataset):
    """MIT-BIH R-peak segmentation dataset."""
    def __init__(self, data_dir, window_sec=5.0, overlap=0.5, sigma=7):
        self.windows = []
        self.targets = []
        records = sorted([f.replace(".dat", "") for f in os.listdir(data_dir) if f.endswith(".dat")])
        logger.info(f"Loading {len(records)} MIT-BIH records for R-peak training...")

        for rec_name in tqdm(records, desc="Loading MIT-BIH"):
            try:
                signal, fs, peaks, _ = load_record_single_lead(os.path.join(data_dir, rec_name))
                # Normalize
                mu, std = np.mean(signal), np.std(signal)
                if std > 1e-6:
                    signal = (signal - mu) / std

                window_samples = int(window_sec * fs)
                stride = int(window_samples * (1 - overlap))

                for start in range(0, len(signal) - window_samples, stride):
                    end = start + window_samples
                    seg = signal[start:end].astype(np.float32)
                    local_peaks = peaks[(peaks >= start) & (peaks < end)] - start

                    # Generate Gaussian target
                    target = np.zeros(window_samples, dtype=np.float32)
                    for p in local_peaks:
                        t = np.arange(window_samples)
                        target = np.maximum(target, np.exp(-0.5 * ((t - p) / sigma) ** 2))

                    self.windows.append(seg)
                    self.targets.append(target)
            except Exception as e:
                logger.warning(f"Skip {rec_name}: {e}")

        logger.info(f"R-peak dataset: {len(self.windows)} windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx]).unsqueeze(0)  # (1, T)
        y = torch.tensor(self.targets[idx]).unsqueeze(0)   # (1, T)
        return x, y


class AFibRhythmDataset(Dataset):
    """MIT-BIH AFDB rhythm classification dataset."""
    def __init__(self, data_dir, window_sec=30.0, target_fs=250):
        self.windows = []
        self.labels = []
        records = sorted([f.replace(".dat", "") for f in os.listdir(data_dir) if f.endswith(".dat")])
        logger.info(f"Loading {len(records)} AFDB records for AFib training...")

        for rec_name in tqdm(records, desc="Loading AFDB"):
            try:
                signal, fs, _, rhythm_labels = load_record_single_lead(
                    os.path.join(data_dir, rec_name), target_fs=target_fs)
                # Normalize
                mu, std = np.mean(signal), np.std(signal)
                if std > 1e-6:
                    signal = (signal - mu) / std

                window_samples = int(window_sec * target_fs)

                for start, end, label in rhythm_labels:
                    label_clean = label.strip("()")
                    is_af = label_clean in ("AFIB", "AFL")

                    # Extract windows from this rhythm segment
                    seg_start = max(0, start)
                    seg_end = min(len(signal), end)

                    for w_start in range(seg_start, seg_end - window_samples, window_samples // 2):
                        w_end = w_start + window_samples
                        if w_end > seg_end:
                            break
                        seg = signal[w_start:w_end].astype(np.float32)
                        self.windows.append(seg)
                        self.labels.append(1 if is_af else 0)

            except Exception as e:
                logger.warning(f"Skip {rec_name}: {e}")

        logger.info(f"AFib dataset: {len(self.windows)} windows, "
                    f"{sum(self.labels)} AF, {len(self.labels)-sum(self.labels)} Normal")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx]).unsqueeze(0)  # (1, T)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ============================================================
# TRAINING LOOPS
# ============================================================

def train_resunet(data_dir, save_path="resunet_rpeak_best.pt"):
    """Train ResU-Net for R-peak segmentation."""
    logger.info("=" * 60)
    logger.info("TRAINING ResU-Net R-Peak Segmentation")
    logger.info("=" * 60)

    dataset = RPeakSegDataset(data_dir, RESUNET_WINDOW_SEC, RESUNET_OVERLAP, GAUSSIAN_SIGMA)

    # Split 90/10
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(SEED))

    train_dl = DataLoader(train_ds, RESUNET_BATCH, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, RESUNET_BATCH, shuffle=False, num_workers=4, pin_memory=True)

    model = ResUNet1D(base=64).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"ResU-Net params: {n_params:,}")

    criterion = FocalDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=RESUNET_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, RESUNET_EPOCHS, RESUNET_MIN_LR)
    scaler = GradScaler()

    best_val_loss = float("inf")
    for epoch in range(RESUNET_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with autocast():
                    pred = model(x)
                    loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_ds)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{RESUNET_EPOCHS} — "
                        f"Train: {train_loss:.5f}, Val: {val_loss:.5f}, LR: {lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ Saved best model (val_loss={val_loss:.5f})")

    logger.info(f"ResU-Net training complete. Best val loss: {best_val_loss:.5f}")
    return save_path


def train_afib_classifier(data_dir, save_path="cnn_transformer_afib_best.pt"):
    """Train CNN-Transformer for AFib classification."""
    logger.info("=" * 60)
    logger.info("TRAINING CNN-Transformer AFib Classifier")
    logger.info("=" * 60)

    dataset = AFibRhythmDataset(data_dir, AFIB_WINDOW_SEC)

    if len(dataset) < 10:
        logger.error("Not enough data for AFib training. Skipping.")
        return None

    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(SEED))

    # Compute class weights for imbalance
    all_labels = [dataset.labels[i] for i in range(len(dataset))]
    n_pos = sum(all_labels)
    n_neg = len(all_labels) - n_pos
    if n_pos > 0 and n_neg > 0:
        weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    else:
        weight = None

    train_dl = DataLoader(train_ds, AFIB_BATCH, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, AFIB_BATCH, shuffle=False, num_workers=4, pin_memory=True)

    model = CNNTransformerAFib(d_model=128, nhead=4, num_layers=3, num_classes=2).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"CNN-Transformer params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=AFIB_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, AFIB_EPOCHS, AFIB_MIN_LR)
    scaler = GradScaler()

    best_val_acc = 0
    for epoch in range(AFIB_EPOCHS):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        train_loss /= total
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{AFIB_EPOCHS} — "
                        f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                        f"Val: loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

    logger.info(f"CNN-Transformer training complete. Best val acc: {best_val_acc:.4f}")
    return save_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    logger.info(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    start = time.time()

    # Download data
    mitdb_dir, afdb_dir = download_datasets()

    # Train ResU-Net on MIT-BIH
    resunet_path = train_resunet(mitdb_dir)

    # Train CNN-Transformer on AFDB
    afib_path = train_afib_classifier(afdb_dir)

    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"ALL TRAINING COMPLETE in {elapsed/60:.1f} minutes")
    logger.info(f"Models saved:")
    logger.info(f"  ResU-Net: {resunet_path}")
    logger.info(f"  CNN-Transformer: {afib_path}")
    logger.info(f"{'='*60}")
    logger.info("\nCopy these .pt files back to your Mac into Heart/models/")
