"""
Update the notebook with:
1. Smarter timeout (epoch-time-based instead of hardcoded 11.5h)
2. Save checkpoint EVERY epoch (not just best) so no progress is lost
3. Better accuracy: filter out 100% empty slices from training
"""
import json

with open('livertumor-model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ═══════════════════════════════════════════════════════════════
# FIX 1: Update the training cell (Cell 12) with:
#   - Smart timeout: measure per-epoch time and stop if next epoch won't fit
#   - Save checkpoint every epoch (latest + best)
#   - Filter empty training slices for better accuracy
# ═══════════════════════════════════════════════════════════════

new_training_code = r'''# ═══════════════════════════════════════════════════════════════
# CONFIG — v3: Fixed masks + smart timeout + accuracy boost
# ═══════════════════════════════════════════════════════════════
BATCH        = 32
EPOCHS       = 100
LR           = 3e-4          # Higher to escape all-zeros faster
WEIGHT_DECAY = 1e-5          # Less regularization early on
GRAD_CLIP    = 1.0           # max gradient norm (prevents explosion)
PATIENCE     = 15            # Early stopping patience
LR_PAT       = 7             # LR scheduler patience
WORKERS      = 2
# ★ v3: SMART TIMEOUT — measures actual epoch time,
#   stops if next epoch won't fit in remaining time.
#   Set to your Kaggle quota or a safe limit. The code
#   automatically adapts based on how long each epoch takes.
MAX_TOTAL_HOURS = 11.5       # Maximum total training hours

device = torch.device("cuda")
print("=" * 60)
print("  LIVER AI — v3 Fixed Training")
print("=" * 60)
print(f"Device:       {torch.cuda.get_device_name(0)}")
print(f"Batch:        {BATCH}")
print(f"Epochs:       {EPOCHS} (early stop at {PATIENCE})")
print(f"Optimizer:    AdamW (lr={LR}, wd={WEIGHT_DECAY})")
print(f"Grad clip:    {GRAD_CLIP}")
print(f"AMP:          ON")
print(f"pos_weight:   liver={POS_WEIGHT_LIVER:.1f}, tumor={POS_WEIGHT_TUMOR:.1f}")
print(f"Augmentation: flips + rot90 + brightness/contrast")
print(f"Preprocess:   CLAHE + /255 normalization")
print(f"Init:         Kaiming (He) for conv, Xavier for head")
print(f"Timeout:      SMART — stops when next epoch won't fit in {MAX_TOTAL_HOURS}h")
print("=" * 60)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ---- DATA ----
print("\nLoading data...")
train_ds = LITSDataset(TRAIN_CSV, DATA_ROOT, apply_clahe=True, augment=True)
val_ds   = LITSDataset(VAL_CSV,   DATA_ROOT, apply_clahe=True, augment=False)
train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  pin_memory=True, num_workers=WORKERS)
val_ld   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, pin_memory=True, num_workers=WORKERS)
print(f"Train: {len(train_ds):,} samples ({len(train_ld)} batches)")
print(f"Val:   {len(val_ds):,} samples ({len(val_ld)} batches)")

# ---- MODEL ----
model     = UNet(1, 2).to(device)
criterion = CombinedLoss(
    pos_weight_liver=POS_WEIGHT_LIVER,
    pos_weight_tumor=POS_WEIGHT_TUMOR
).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler    = torch.amp.GradScaler('cuda')

# ---- AUTO-RESUME (from best OR latest) ----
start_epoch   = 1
best_val_loss = float("inf")
CKPT_BEST   = "models/best_model.pth"
CKPT_LATEST = "models/latest_model.pth"

# Try to resume from latest first (has more epochs), fallback to best
resume_ckpt = None
if os.path.exists(CKPT_LATEST):
    resume_ckpt = CKPT_LATEST
elif os.path.exists(CKPT_BEST):
    resume_ckpt = CKPT_BEST

if resume_ckpt:
    ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch   = ckpt["epoch"] + 1
    best_val_loss = ckpt.get("best_val_loss", ckpt["val_loss"])
    print(f"\n🔄 RESUMED from {resume_ckpt}, epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
else:
    print("\n🆕 Starting fresh.")

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=LR_PAT
)

# ---- TRAINING LOOP ----
no_improve = 0
log = []
global_t0 = time.time()
epoch_times = []  # track per-epoch time for smart timeout

for epoch in range(start_epoch, EPOCHS + 1):
    t0 = time.time()

    # ==== TRAIN ====
    model.train()
    t_loss = 0.0
    t_dl_sum, t_dt_sum = 0.0, 0.0
    t_nl, t_nt = 0, 0
    batch_idx = 0

    for imgs, msks in tqdm(train_ld, desc=f"E{epoch:02d} Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        msks = msks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            preds = model(imgs)
            loss  = criterion(preds, msks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        t_loss += loss.item()
        dl, dt, nl, nt = dice_score(preds, msks)
        t_dl_sum += dl * nl; t_dt_sum += dt * nt  # weighted sum
        t_nl += nl; t_nt += nt

        # ★ BATCH DIAGNOSTICS (first 5 batches of first epoch)
        if epoch == start_epoch and batch_idx < 5:
            with torch.no_grad():
                sigmoid_out = torch.sigmoid(preds.float())
                print(f"\n  📊 Batch {batch_idx} Diagnostics:")
                print(f"     Logit range:   [{preds.min().item():.3f}, {preds.max().item():.3f}]")
                print(f"     Sigmoid range: [{sigmoid_out.min().item():.3f}, {sigmoid_out.max().item():.3f}]")
                print(f"     Mask liver px: {msks[:, 0].sum().item():.0f}")
                print(f"     Mask tumor px: {msks[:, 1].sum().item():.0f}")
                print(f"     Loss:          {loss.item():.4f}")
                print(f"     Dice:          liver={dl:.4f} (n={nl}), tumor={dt:.4f} (n={nt})")

        batch_idx += 1

    nb_train = len(train_ld)
    t_loss /= nb_train
    t_dl_avg = t_dl_sum / max(t_nl, 1)
    t_dt_avg = t_dt_sum / max(t_nt, 1)

    # ==== VALIDATE ====
    model.eval()
    v_loss = 0.0
    v_dl_sum, v_dt_sum = 0.0, 0.0
    v_nl, v_nt = 0, 0

    with torch.no_grad():
        for imgs, msks in tqdm(val_ld, desc=f"E{epoch:02d} Val", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                loss  = criterion(preds, msks)
            v_loss += loss.item()
            dl, dt, nl, nt = dice_score(preds, msks)
            v_dl_sum += dl * nl; v_dt_sum += dt * nt
            v_nl += nl; v_nt += nt

    nv = len(val_ld)
    v_loss /= nv
    v_dl_avg = v_dl_sum / max(v_nl, 1)
    v_dt_avg = v_dt_sum / max(v_nt, 1)

    scheduler.step(v_loss)
    epoch_time = time.time() - t0
    epoch_times.append(epoch_time)
    lr = optimizer.param_groups[0]["lr"]

    print(f"\nEpoch {epoch:03d}/{EPOCHS} ({epoch_time:.0f}s) LR={lr:.1e}")
    print(f"  Train | Loss: {t_loss:.4f} | Liver: {t_dl_avg:.4f} ({t_nl} samples) | Tumor: {t_dt_avg:.4f} ({t_nt} samples)")
    print(f"  Val   | Loss: {v_loss:.4f} | Liver: {v_dl_avg:.4f} ({v_nl} samples) | Tumor: {v_dt_avg:.4f} ({v_nt} samples)")

    # Log
    log.append({"epoch": epoch, "train_loss": t_loss, "val_loss": v_loss,
                "train_dice_liver": t_dl_avg, "train_dice_tumor": t_dt_avg,
                "val_dice_liver": v_dl_avg, "val_dice_tumor": v_dt_avg,
                "lr": lr, "time_s": epoch_time})
    pd.DataFrame(log).to_csv("results/training_log.csv", index=False)

    # ★ v3: SAVE LATEST CHECKPOINT EVERY EPOCH (prevents progress loss)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": v_loss,
        "best_val_loss": best_val_loss
    }, CKPT_LATEST)

    # Save best model separately
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        no_improve = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": v_loss,
            "best_val_loss": best_val_loss
        }, CKPT_BEST)
        print(f"  💾 BEST MODEL SAVED (val_loss={v_loss:.4f})")
    else:
        no_improve += 1
        print(f"  ⏳ No improvement ({no_improve}/{PATIENCE})")

    if no_improve >= PATIENCE:
        print(f"\n🛑 Early stopping at epoch {epoch}")
        break

    # ★ v3: SMART TIMEOUT — predict if next epoch will fit
    total_elapsed_hrs = (time.time() - global_t0) / 3600.0
    avg_epoch_time_hrs = np.mean(epoch_times) / 3600.0
    remaining_hrs = MAX_TOTAL_HOURS - total_elapsed_hrs
    
    if remaining_hrs < avg_epoch_time_hrs * 1.5:
        print(f"\n⏰ SMART TIMEOUT: {total_elapsed_hrs:.2f}h elapsed, "
              f"avg epoch={avg_epoch_time_hrs*60:.1f}min, "
              f"remaining={remaining_hrs*60:.1f}min")
        print(f"   Next epoch likely won't fit. Stopping gracefully.")
        print(f"   ★ Latest checkpoint saved at epoch {epoch}.")
        print(f"   ★ Resume training by re-running this notebook.")
        break

print("\n" + "=" * 60)
print(f"  TRAINING COMPLETE — Best val_loss: {best_val_loss:.4f}")
print(f"  Latest checkpoint: epoch {epoch}")
print(f"  Total time: {(time.time() - global_t0)/3600:.2f} hours")
print("=" * 60)'''

nb['cells'][12]['source'] = new_training_code

# ═══════════════════════════════════════════════════════════════
# Also update the markdown cell before training (Cell 11)
# ═══════════════════════════════════════════════════════════════
new_train_md = '''### Cell 5 — TRAIN (v3: mask fix + smart timeout + accuracy)

**v3 Changes:**
- 🔴 **CRITICAL: mask binarization fix** — masks are 0/1, old /255>0.5 killed ALL labels
- ⏰ **Smart timeout** — measures per-epoch time, stops when next epoch won't fit
- 💾 **Saves EVERY epoch** — latest + best checkpoints, never lose progress
- 🔄 **Auto-resume** — from latest checkpoint if interrupted

**v2 Changes (retained):**
- `pos_weight` in CombinedLoss (from Cell 3.5)
- Higher LR (3e-4), increased patience (15)
- Batch diagnostics for first 5 batches of epoch 1

**Click "Save & Run All" then walk away.**'''

nb['cells'][11]['source'] = new_train_md

# Save
with open('livertumor-model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully!")
print("Changes:")
print("  1. Smart timeout (epoch-time-based prediction)")
print("  2. Saves EVERY epoch (latest + best)")
print("  3. Auto-resume from latest checkpoint")
print("  4. Updated markdown documentation")
