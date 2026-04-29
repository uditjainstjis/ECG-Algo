"""
CNN-Transformer Hybrid for AFib Classification
=================================================
CNN frontend extracts local morphology features, Transformer encoder
captures long-range rhythm irregularity, attention pooling aggregates
into a fixed embedding for classification.

Input:  30s raw ECG window (B, 1, 7500) @ 250Hz
Output: AF probability (B, 1) or multi-class logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding1D(nn.Module):
    """Sinusoidal positional encoding for token sequences."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class AttentionPooling(nn.Module):
    """Learned attention pooling over token sequence."""

    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return (x * weights).sum(dim=1)  # (B, D)


class CNNTransformerAFib(nn.Module):
    """
    CNN-Transformer hybrid for AF classification from raw ECG.

    Pipeline:
        Raw ECG → CNN stem (local features) → Token sequence →
        Positional encoding → Transformer encoder (rhythm context) →
        Attention pooling → Classification head
    """

    def __init__(
        self,
        in_ch: int = 1,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        num_classes: int = 1,
    ):
        super().__init__()

        # CNN frontend: downsample 8x while extracting local features
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, d_model, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding1D(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling + classifier
        self.pool = AttentionPooling(d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T) raw ECG

        Returns:
            (B, num_classes) logits
        """
        # CNN: (B, 1, T) → (B, d_model, T')
        feat = self.cnn(x)

        # Reshape to token sequence: (B, T', d_model)
        tokens = feat.transpose(1, 2)

        # Add positional encoding
        tokens = self.pos_enc(tokens)

        # Transformer: inter-beat context
        tokens = self.transformer(tokens)

        # Pool and classify
        embedding = self.pool(tokens)
        return self.classifier(embedding)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Extract attention pooling weights for interpretability."""
        feat = self.cnn(x)
        tokens = feat.transpose(1, 2)
        tokens = self.pos_enc(tokens)
        tokens = self.transformer(tokens)
        weights = torch.softmax(self.pool.attn(tokens), dim=1)
        return weights.squeeze(-1)
