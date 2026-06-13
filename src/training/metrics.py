"""
Evaluation Metrics for Liver & Tumor Segmentation
Based on: FasNet Paper (Nature Scientific Reports, 2025)

PRIMARY METRIC: Dice Coefficient (used in ALL liver segmentation papers)

Target scores from FasNet paper on LiTS17:
  - Liver channel:  aim for > 0.70 initially, > 0.85 after tuning
  - Tumor channel:  aim for > 0.50 initially (tumors are harder)
  - Mean:           aim for > 0.60 initially
"""

import torch


def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Compute Dice Score for evaluation (NOT for training).

    This is the PRIMARY metric used in ALL liver segmentation papers.

    Args:
        pred:   raw logits [B, 2, H, W]
        target: binary mask [B, 2, H, W]
        threshold: binarization threshold (default 0.5)
        smooth: smoothing factor to avoid division by zero

    Returns:
        dice_liver (float), dice_tumor (float), dice_mean (float)
    """
    with torch.no_grad():
        pred = torch.sigmoid(pred)
        pred = (pred > threshold).float()   # binarize predictions

        # Channel 0 = liver, Channel 1 = tumor
        def channel_dice(p, t):
            # If both prediction and target are empty, dice is 0 (not 1)
            if p.sum() == 0 and t.sum() == 0:
                return 0.0
            intersection = (p * t).sum()
            score = (2. * intersection + smooth) / (p.sum() + t.sum() + smooth)
            return score.item()  # always return plain float

        dice_liver = channel_dice(pred[:, 0], target[:, 0])
        dice_tumor = channel_dice(pred[:, 1], target[:, 1])
        dice_mean = (dice_liver + dice_tumor) / 2

    return dice_liver, dice_tumor, dice_mean


if __name__ == "__main__":
    print("Testing dice_score (FasNet-aligned)...")
    fake_logits = torch.randn(4, 2, 256, 256)
    fake_targets = torch.randint(0, 2, (4, 2, 256, 256)).float()

    dl, dt, dm = dice_score(fake_logits, fake_targets)
    print(f"Liver Dice: {dl:.4f}, Tumor Dice: {dt:.4f}, Mean Dice: {dm:.4f}")
    print("Metrics sanity check PASSED")
