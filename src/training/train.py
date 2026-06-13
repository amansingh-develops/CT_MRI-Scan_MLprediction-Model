"""
Training Pipeline for Liver & Tumor Segmentation
Based on: FasNet Paper (Nature Scientific Reports, 2025)
DOI: 10.1038/s41598-025-98427-9

PROVEN CONFIG from FasNet Paper (do NOT change without reason):
  - Optimizer: Adam (Dice 0.8766) >> SGD (0.7913), RMSProp, AdaGrad, AdaDelta
  - Batch size: 32 >> 16, 64, 128 on Dice score
  - Epochs: 100 (model stabilizes ~epoch 60)
  - Learning rate: 1e-4 (standard for Adam on medical segmentation)
  - Loss: Combined (0.5 BCE + 0.5 Dice)
"""

import os
import time
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our custom modules
from src.data.dataset_loader import LITSDataset
from src.models.unet import UNet
from src.training.loss import CombinedLoss
from src.training.validate import validate
from src.training.metrics import dice_score
from src.utils.checkpoint import save_checkpoint


# ═══════════════════════════════════════════════════════════════
# CONFIG — ALL VALUES PROVEN BY FASNET PAPER (DO NOT CHANGE)
# ═══════════════════════════════════════════════════════════════
CONFIG = {
    # Paths
    "train_csv":       "/kaggle/input/datasets/andrewmvd/lits-png/lits_train.csv",
    "val_csv":         "/kaggle/input/datasets/andrewmvd/lits-png/lits_probe.csv",   # probe set for validation per PRD
    "checkpoint_dir":  "models/",
    "results_dir":     "results/",

    # Model
    "in_channels":     1,
    "out_channels":    2,
    "image_size":      256,

    # Training — ALL PROVEN BY FASNET PAPER
    "batch_size":      32,       # PAPER RESULT: batch 32 > 16, 64, 128 on Dice score
    "num_epochs":      100,      # PAPER RESULT: model stabilizes ~epoch 60, run to 100
    "learning_rate":   1e-4,     # Standard for Adam on medical segmentation

    # Anti-overfitting
    "early_stopping_patience": 10,      # stop if val_loss doesn't improve for 10 epochs
    "lr_scheduler_patience":   5,       # halve LR if no improvement for 5 epochs
    "lr_scheduler_factor":     0.5,

    # Hardware
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers":     2,
    "pin_memory":      True,     # speeds up GPU data transfer

    # Preprocessing
    "apply_clahe":     True,     # CLAHE applied dynamically in DataLoader
}


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    One full pass through the training data.
    Returns: avg_loss, avg_dice_liver, avg_dice_tumor
    """
    model.train()
    total_loss = 0
    total_dice_liver = 0
    total_dice_tumor = 0

    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Training", leave=False)):
        # Step 1: Move data to GPU
        images = images.to(device)
        masks = masks.to(device)

        # Step 2: Forward pass
        preds = model(images)               # [B, 2, 256, 256] — raw logits

        # Step 3: Compute loss
        loss = criterion(preds, masks)      # CombinedLoss(logits, binary_mask)

        # Step 4: Backward pass
        optimizer.zero_grad()               # clear old gradients
        loss.backward()                     # compute new gradients
        optimizer.step()                    # update weights

        # Step 5: Track metrics
        total_loss += loss.item()
        dl, dt, _ = dice_score(preds, masks)
        total_dice_liver += dl
        total_dice_tumor += dt

    n = len(loader)
    if n == 0:
        return 0, 0, 0
    return total_loss / n, total_dice_liver / n, total_dice_tumor / n


def main():
    device = torch.device(CONFIG["device"])
    print(f"=== Liver AI Training (FasNet-aligned) ===")
    print(f"Device: {device}")
    print(f"Batch size: {CONFIG['batch_size']} (paper-proven optimal)")
    print(f"Epochs: {CONFIG['num_epochs']} (paper shows stability ~epoch 60)")
    print(f"Optimizer: Adam (paper-proven best: Dice 0.8766)")

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    # ── 1. Datasets & DataLoaders ──
    print("\nLoading datasets...")
    
    # Hardcoded path — images are here, period.
    data_root = "/kaggle/input/datasets/andrewmvd/lits-png/dataset_6/dataset_6"

    print(f"[INFO] Using data_root: {data_root}")
        
    train_dataset = LITSDataset(
        CONFIG["train_csv"], root_dir=data_root, apply_clahe=CONFIG["apply_clahe"]
    )
    val_dataset = LITSDataset(
        CONFIG["val_csv"], root_dir=data_root, apply_clahe=CONFIG["apply_clahe"]
    )

    num_w = CONFIG["num_workers"] if CONFIG["device"] == "cuda" else 0
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        pin_memory=CONFIG["pin_memory"], num_workers=num_w
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
        pin_memory=CONFIG["pin_memory"], num_workers=num_w
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ── 2. Model, Loss, Optimizer, Scheduler ──
    model = UNet(in_channels=1, out_channels=2).to(device)
    criterion = CombinedLoss()

    # PAPER PROVEN: Adam optimizer is best for this task
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # ── 2.5 Auto-Resume from Checkpoint ──
    start_epoch = 1
    best_val_loss = float("inf")
    checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"\n[INFO] Found existing checkpoint at {checkpoint_path}. Resuming training!")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]
        print(f"[INFO] Resuming from Epoch {start_epoch} with previous best val_loss={best_val_loss:.4f}\n")

    # Reduce LR when validation plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=CONFIG["lr_scheduler_factor"],
        patience=CONFIG["lr_scheduler_patience"]
    )

    # ── 3. Training Loop ──
    no_improve_count = 0
    log_rows = []
    log_path = os.path.join(CONFIG["results_dir"], "training_log.csv")

    for epoch in range(start_epoch, CONFIG["num_epochs"] + 1):
        start_time = time.time()

        train_loss, train_dl, train_dt = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_dl, val_dt = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # Print Epoch Summary
        print(f"\nEpoch {epoch:03d}/{CONFIG['num_epochs']} ({epoch_time:.0f}s)")
        print(f"  Train → Loss: {train_loss:.4f} | Dice Liver: {train_dl:.4f} | Dice Tumor: {train_dt:.4f}")
        print(f"  Val   → Loss: {val_loss:.4f}   | Dice Liver: {val_dl:.4f}   | Dice Tumor: {val_dt:.4f}")
        print(f"  LR: {current_lr:.2e}")

        # Save log
        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_dice_liver": train_dl, "train_dice_tumor": train_dt,
            "val_dice_liver": val_dl, "val_dice_tumor": val_dt,
            "lr": current_lr
        })
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        # ── 4. Checkpointing & Early Stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")
            )
            print(f"  >> Best model saved (val_loss={val_loss:.4f})")
        else:
            no_improve_count += 1
            print(f"  No improvement for {no_improve_count} epoch(s)")

        # Early stopping
        if no_improve_count >= CONFIG["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Best val_loss: {best_val_loss:.4f}")
            break

    print("\n" + "=" * 50)
    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Target from FasNet paper: Dice ~0.87 (after 100 epochs with attention)")
    print(f"Your base U-Net target:   Dice ~0.70-0.77 (reasonable first run)")
    print("=" * 50)


if __name__ == "__main__":
    main()
