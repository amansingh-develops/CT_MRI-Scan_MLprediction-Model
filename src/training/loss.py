"""
Loss Functions for Liver & Tumor Segmentation
Based on: FasNet Paper (Nature Scientific Reports, 2025)

PAPER EVIDENCE: All top methods on LiTS17 use Dice-based losses.
Combined Loss = 0.5 * BCE + 0.5 * Dice is the standard.
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.

    WHY: Most CT slices have very few tumor pixels. BCE alone ignores rare pixels.
         Dice loss focuses on OVERLAP, so rare classes matter more.

    Formula: 1 - (2 * |intersection| + smooth) / (|pred| + |target| + smooth)
    smooth = 1e-6 to avoid division by zero

    Input: pred   = raw logits [B, 2, H, W]  — NO sigmoid yet
           target = binary mask [B, 2, H, W]  — values are 0 or 1
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)              # convert logits to 0-1 range
        pred = pred.contiguous()
        target = target.contiguous()

        # Flatten spatial dimensions for computation — sum over H and W
        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (
            pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth
        )
        # dice shape: [B, 2] — one score per channel per sample
        return 1 - dice.mean()                  # return scalar loss


class CombinedLoss(nn.Module):
    """
    CombinedLoss = 0.5 * BCE + 0.5 * Dice

    WHY COMBINE:
    - BCE: good at learning from individual pixel classifications
    - Dice: good at handling class imbalance (few tumor pixels vs background)
    - Together: more stable training, better final Dice score

    PAPER EVIDENCE: All top methods on LiTS17 use Dice-based losses.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()    # takes raw logits, applies sigmoid internally
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return 0.5 * bce_loss + 0.5 * dice_loss


if __name__ == "__main__":
    print("Testing CombinedLoss (FasNet-aligned)...")
    loss_fn = CombinedLoss()
    fake_logits = torch.randn(4, 2, 256, 256)
    fake_targets = torch.randint(0, 2, (4, 2, 256, 256)).float()

    loss = loss_fn(fake_logits, fake_targets)
    print(f"Computed combined loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    print("Loss sanity check PASSED")
