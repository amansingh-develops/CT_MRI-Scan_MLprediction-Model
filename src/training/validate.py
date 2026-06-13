"""
Validation loop for Liver & Tumor Segmentation
Based on: FasNet Paper (Nature Scientific Reports, 2025)
"""

import torch
from src.training.metrics import dice_score
from tqdm import tqdm


def validate(model, loader, criterion, device):
    """
    One full pass through the validation data. No gradient computation.
    Returns: avg_loss, avg_dice_liver, avg_dice_tumor
    """
    model.eval()
    total_loss = 0
    total_dice_liver = 0
    total_dice_tumor = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = criterion(preds, masks)

            total_loss += loss.item()
            dl, dt, _ = dice_score(preds, masks)
            total_dice_liver += dl
            total_dice_tumor += dt

    n = len(loader)
    if n == 0:
        return 0, 0, 0

    return total_loss / n, total_dice_liver / n, total_dice_tumor / n
