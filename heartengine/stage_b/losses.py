"""
Focal + Dice Composite Loss
==============================
For R-peak segmentation with extreme class imbalance (~0.3% positive).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Focal loss for binary segmentation. Downweights easy examples."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(1e-7, 1 - 1e-7)
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        return (alpha_weight * focal_weight * bce).mean()


class SoftDiceLoss(nn.Module):
    """Soft Dice loss for 1D segmentation masks."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class FocalDiceLoss(nn.Module):
    """Composite: λ_f * Focal + λ_d * Dice."""

    def __init__(
        self,
        focal_weight: float = 0.7,
        dice_weight: float = 0.3,
        gamma: float = 2.0,
        alpha: float = 0.75,
    ):
        super().__init__()
        self.focal = BinaryFocalLoss(gamma=gamma, alpha=alpha)
        self.dice = SoftDiceLoss()
        self.fw = focal_weight
        self.dw = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.fw * self.focal(pred, target) + self.dw * self.dice(pred, target)
