"""
v6 "Ironclad" Patch — Applies all 6 bug fixes to Cell 12 of livertumor-model.ipynb

Fixes:
  1. GRAD_CLIP 1.0 → 5.0
  2. Warmup always relative to start_epoch  
  3. Inf gradient handler: count + auto-disable AMP
  4. Catastrophic loss detection + auto-rollback
  5. get_warmup_lr relative to start_epoch
  6. no_improve counter reset after rollback
"""
import json, sys, os, shutil

sys.stdout.reconfigure(encoding='utf-8')

NB_PATH = r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\livertumor-model.ipynb'

# ── Load notebook ──
with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── Backup ──
backup = NB_PATH + '.v5_backup'
if not os.path.exists(backup):
    shutil.copy2(NB_PATH, backup)
    print(f"✅ Backup saved: {backup}")
else:
    print(f"ℹ️  Backup already exists: {backup}")

# ── Get Cell 12 source as a single string ──
cell12_lines = nb['cells'][12]['source']
cell12 = ''.join(cell12_lines)

# ═══════════════════════════════════════════════════════════════
# Verify we're editing the right cell
# ═══════════════════════════════════════════════════════════════
assert 'CONFIG — v5' in cell12 or 'GRAD_CLIP' in cell12, \
    "Cell 12 doesn't look like the training cell!"
print("✅ Cell 12 identified as training cell")

# ═══════════════════════════════════════════════════════════════
# BUILD THE COMPLETE NEW CELL 12 — v6 "Ironclad"
# ═══════════════════════════════════════════════════════════════

NEW_CELL_12 = r'''# ═══════════════════════════════════════════════════════════════
# CONFIG — v6: Ironclad Training Pipeline
# ═══════════════════════════════════════════════════════════════
BATCH        = 32            # PAPER: batch 32 > 16, 64, 128
EPOCHS       = 100           # PAPER: model stabilizes ~epoch 60
LR           = 1e-4          # PAPER: standard for Adam on medical segmentation
GRAD_CLIP    = 5.0           # FIX #2: healthy norms are 7-12, clip=1.0 was crushing ALL gradients
PATIENCE     = 10            # PAPER: early stopping patience
LR_PAT       = 5             # PAPER: LR scheduler patience
LR_FACTOR    = 0.5           # PAPER: halve LR on plateau
WORKERS      = 2
MAX_TOTAL_HOURS = 11.5       # Kaggle GPU time limit safety
WARMUP_EPOCHS = 3            # v6: warmup 3 epochs from any start point
MAX_INF_PER_EPOCH = 5        # v6: if >5 Inf batches, switch to FP32 mid-epoch

# ★ RESUME: Set this to the path of your previous output dataset.
#   Example: "/kaggle/input/your-previous-output-dataset/models/best_model.pth"
#   Leave as None to auto-detect.
PREV_CHECKPOINT = None

device = torch.device("cuda")
print("=" * 60)
print("  LIVER AI — v6 Ironclad Training Pipeline")
print("=" * 60)
print(f"Device:       {torch.cuda.get_device_name(0)}")
print(f"Batch:        {BATCH} (paper-proven)")
print(f"Epochs:       {EPOCHS} (early stop at {PATIENCE})")
print(f"Optimizer:    Adam (lr={LR}) — paper-proven")
print(f"Loss:         0.5*BCE + 0.5*Dice — NO pos_weight")
print(f"Grad clip:    {GRAD_CLIP} (healthy norms are 7-12)")
print(f"AMP:          ON (with Inf cascade protection)")
print(f"Warmup:       {WARMUP_EPOCHS} epochs from start_epoch")
print(f"LR Scheduler: ReduceLR(patience={LR_PAT}, factor={LR_FACTOR})")
print(f"Timeout:      SMART — stops when next epoch won't fit in {MAX_TOTAL_HOURS}h")
print("=" * 60)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ---- DATA ----
print("\nLoading data...")
train_ds = LITSDataset(TRAIN_CSV, DATA_ROOT, augment=True)
val_ds   = LITSDataset(VAL_CSV,   DATA_ROOT, augment=False)
train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  pin_memory=True, num_workers=WORKERS)
val_ld   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, pin_memory=True, num_workers=WORKERS)
print(f"Train: {len(train_ds):,} samples ({len(train_ld)} batches)")
print(f"Val:   {len(val_ds):,} samples ({len(val_ld)} batches)")

# ---- MODEL + LOSS + OPTIMIZER (paper-aligned) ----
model     = UNet(1, 2).to(device)
criterion = CombinedLoss().to(device)  # plain BCE + Dice, NO pos_weight
optimizer = optim.Adam(model.parameters(), lr=LR)  # plain Adam, paper-proven
scaler    = torch.amp.GradScaler('cuda')

# ── CHECKPOINT SEARCH ─────────────────────────────────────────
CKPT_BEST   = "models/best_model.pth"
CKPT_LATEST = "models/latest_model.pth"

start_epoch   = 1
best_val_loss = float("inf")
weights_only_resume = False

def find_prev_checkpoint():
    """Search /kaggle/input/ for checkpoints from previous notebook output."""
    if PREV_CHECKPOINT and os.path.exists(PREV_CHECKPOINT):
        return PREV_CHECKPOINT
    patterns = [
        "/kaggle/input/*/models/best_model.pth",
        "/kaggle/input/*/*/models/best_model.pth",
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None

# Try current working dir first
resume_ckpt = None
if os.path.exists(CKPT_LATEST):
    resume_ckpt = CKPT_LATEST
    weights_only_resume = False
elif os.path.exists(CKPT_BEST):
    resume_ckpt = CKPT_BEST
    weights_only_resume = False
else:
    prev = find_prev_checkpoint()
    if prev:
        resume_ckpt = prev
        weights_only_resume = True

if resume_ckpt:
    print(f"\n📂 Found checkpoint: {resume_ckpt}")
    ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    
    if weights_only_resume:
        print(f"   🔄 WEIGHTS-ONLY resume (from previous version)")
        print(f"   Previous epoch: {ckpt['epoch']}, val_loss: {ckpt['val_loss']:.4f}")
        print(f"   ⚠️ Optimizer and scaler reset — fresh start with pretrained weights")
        print(f"   ✅ Warmup will apply for {WARMUP_EPOCHS} epochs from epoch 1")
    else:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # v6: ALSO restore GradScaler state
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            print(f"   ✅ GradScaler state restored")
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", ckpt["val_loss"])
        print(f"   🔄 FULL resume from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        print(f"   ✅ Warmup will apply for {WARMUP_EPOCHS} epochs from epoch {start_epoch}")
else:
    print("\n🆕 Starting fresh — no checkpoint found.")

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=LR_FACTOR, patience=LR_PAT
)

# ═══════════════════════════════════════════════════════════════
# FIX #5: WARMUP relative to start_epoch, not absolute
# ═══════════════════════════════════════════════════════════════
def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr

def get_warmup_lr(current_epoch, start_ep, base_lr, warmup_epochs):
    """Linear warmup relative to start_epoch.
    
    epoch=start_ep   → base_lr * 0.1
    epoch=start_ep+2 → base_lr * 1.0  (for warmup_epochs=3)
    """
    relative = current_epoch - start_ep + 1  # 1, 2, 3...
    if relative > warmup_epochs:
        return base_lr
    return base_lr * (0.1 + 0.9 * (relative - 1) / max(1, warmup_epochs - 1))

# ── TRAINING LOOP ─────────────────────────────────────────────
no_improve    = 0
log           = []
global_t0     = time.time()
epoch_times   = []
nan_recovery  = False    # v6: True if we just recovered from NaN
use_amp       = True     # v6: can be disabled during recovery
warmup_end_epoch = start_epoch + WARMUP_EPOCHS - 1  # v6: warmup ends here

print(f"\n🎯 Training plan: epochs {start_epoch}→{EPOCHS}")
print(f"   Warmup: epochs {start_epoch}→{warmup_end_epoch}")
print(f"   Scheduler active after epoch {warmup_end_epoch}")

for epoch in range(start_epoch, EPOCHS + 1):
    t0 = time.time()

    # ═══════════════════════════════════════════════════════════
    # FIX #2: WARMUP always runs for WARMUP_EPOCHS from start
    # ═══════════════════════════════════════════════════════════
    if epoch <= warmup_end_epoch:
        warmup_lr = get_warmup_lr(epoch, start_epoch, LR, WARMUP_EPOCHS)
        set_lr(optimizer, warmup_lr)
        print(f"  🔥 Warmup: lr={warmup_lr:.2e}")

    # v6: If recovering from NaN/catastrophe, disable AMP for this epoch
    if nan_recovery:
        use_amp = False
        print(f"  🛡️ AMP DISABLED for recovery epoch (fp32 mode)")
    else:
        use_amp = True

    # ==== TRAIN ====
    model.train()
    t_loss = 0.0
    t_dl_sum, t_dt_sum = 0.0, 0.0
    n_batches_ok = 0
    nan_in_epoch = False
    max_grad_norm = 0.0
    inf_count_this_epoch = 0  # v6 FIX #3: Inf cascade counter

    for imgs, msks in tqdm(train_ld, desc=f"E{epoch:02d} Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        msks = msks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # v6: conditional AMP
        if use_amp:
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                loss  = criterion(preds, msks)
        else:
            preds = model(imgs)
            loss  = criterion(preds, msks)

        # ★ NaN DETECTION
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n  ⚠️ NaN/Inf loss at epoch {epoch}, skipping batch")
            nan_in_epoch = True
            optimizer.zero_grad(set_to_none=True)
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            
            # ═══════════════════════════════════════════════════
            # FIX #3: Inf cascade protection
            # ═══════════════════════════════════════════════════
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                inf_count_this_epoch += 1
                # DON'T call scaler.step() — just update() to adjust scale
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                if inf_count_this_epoch <= 3:
                    print(f"\n  ⚠️ Inf gradient norm ({inf_count_this_epoch}/{MAX_INF_PER_EPOCH}), scaler skip")
                
                if inf_count_this_epoch >= MAX_INF_PER_EPOCH:
                    print(f"\n  🛡️ Inf cascade detected ({inf_count_this_epoch} events) — disabling AMP for rest of epoch")
                    use_amp = False
                    # Reset scaler for next epoch
                    scaler = torch.amp.GradScaler('cuda')
                continue
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

        t_loss += loss.item()
        if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
            max_grad_norm = max(max_grad_norm, grad_norm.item())
        dl, dt, _ = dice_score(preds, msks)
        t_dl_sum += dl
        t_dt_sum += dt
        n_batches_ok += 1

    # v6: Handle full-epoch NaN
    if n_batches_ok == 0:
        print(f"\n💀 ALL batches had NaN loss in epoch {epoch}!")
        print(f"   🔄 Resetting GradScaler and loading best checkpoint...")
        # v6: FULL RECOVERY — reset scaler, load best model
        scaler = torch.amp.GradScaler('cuda')  # fresh scaler
        if os.path.exists(CKPT_BEST):
            ckpt = torch.load(CKPT_BEST, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer = optim.Adam(model.parameters(), lr=LR * 0.5)  # lower LR
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=LR_FACTOR, patience=LR_PAT)
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            no_improve = 0  # FIX #6: reset patience
            print(f"   ✅ Rolled back to best model (epoch {ckpt['epoch']})")
            print(f"   ✅ Fresh optimizer (lr={LR*0.5:.1e}) + scaler + scheduler")
        nan_recovery = True  # next epoch runs in fp32
        continue
    else:
        nan_recovery = False  # successful epoch, resume AMP

    t_loss   /= n_batches_ok
    t_dl_avg = t_dl_sum / n_batches_ok
    t_dt_avg = t_dt_sum / n_batches_ok

    # ==== VALIDATE ====
    model.eval()
    v_loss = 0.0
    v_dl_sum, v_dt_sum = 0.0, 0.0
    n_val = 0

    with torch.no_grad():
        for imgs, msks in tqdm(val_ld, desc=f"E{epoch:02d} Val", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                loss  = criterion(preds, msks)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                v_loss += loss.item()
                dl, dt, _ = dice_score(preds, msks)
                v_dl_sum += dl
                v_dt_sum += dt
                n_val += 1

    if n_val > 0:
        v_loss   /= n_val
        v_dl_avg = v_dl_sum / n_val
        v_dt_avg = v_dt_sum / n_val
    else:
        v_loss = float('inf')
        v_dl_avg = 0.0
        v_dt_avg = 0.0
        print(f"  ⚠️ All validation batches had NaN loss!")

    # ═══════════════════════════════════════════════════════════
    # FIX #4: CATASTROPHIC SPIKE DETECTION + AUTO-ROLLBACK
    # ═══════════════════════════════════════════════════════════
    if v_loss > best_val_loss * 2.0 and best_val_loss < float('inf'):
        print(f"\n  🚨 CATASTROPHIC SPIKE: val_loss={v_loss:.4f} > 2× best={best_val_loss:.4f}")
        print(f"  🔄 Auto-rolling back to best checkpoint...")
        if os.path.exists(CKPT_BEST):
            ckpt = torch.load(CKPT_BEST, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            # Use a lower LR after catastrophe to prevent recurrence
            recovery_lr = LR * 0.5
            optimizer = optim.Adam(model.parameters(), lr=recovery_lr)
            scaler = torch.amp.GradScaler('cuda')
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=LR_FACTOR, patience=LR_PAT)
            no_improve = 0  # FIX #6: reset patience counter
            nan_recovery = True  # next epoch runs in fp32 for safety
            print(f"  ✅ Rolled back to best model (epoch {ckpt['epoch']})")
            print(f"  ✅ Recovery LR={recovery_lr:.1e}, next epoch in FP32")
            # Log the spike but continue
            log.append({"epoch": epoch, "train_loss": t_loss, "val_loss": v_loss,
                        "train_dice_liver": t_dl_avg, "train_dice_tumor": t_dt_avg,
                        "val_dice_liver": v_dl_avg, "val_dice_tumor": v_dt_avg,
                        "lr": optimizer.param_groups[0]["lr"],
                        "time_s": time.time() - t0, "max_grad_norm": max_grad_norm,
                        "amp": use_amp, "event": "CATASTROPHIC_ROLLBACK"})
            pd.DataFrame(log).to_csv("results/training_log.csv", index=False)
            continue
        else:
            print(f"  ⚠️ No best checkpoint found to rollback to!")

    # Only step scheduler after warmup
    if epoch > warmup_end_epoch:
        scheduler.step(v_loss)
    
    epoch_time = time.time() - t0
    epoch_times.append(epoch_time)
    lr = optimizer.param_groups[0]["lr"]

    print(f"\nEpoch {epoch:03d}/{EPOCHS} ({epoch_time:.0f}s) LR={lr:.1e}")
    print(f"  Train | Loss: {t_loss:.4f} | Liver: {t_dl_avg:.4f} | Tumor: {t_dt_avg:.4f}")
    print(f"  Val   | Loss: {v_loss:.4f} | Liver: {v_dl_avg:.4f} | Tumor: {v_dt_avg:.4f}")
    print(f"  Grad  | Max norm: {max_grad_norm:.4f} | AMP: {'ON' if use_amp else 'OFF (recovery)'}")
    if inf_count_this_epoch > 0:
        print(f"  ⚠️ {inf_count_this_epoch} Inf gradient events this epoch")
    if nan_in_epoch:
        print(f"  ⚠️ Some batches had NaN loss this epoch (skipped)")

    # Log
    log.append({"epoch": epoch, "train_loss": t_loss, "val_loss": v_loss,
                "train_dice_liver": t_dl_avg, "train_dice_tumor": t_dt_avg,
                "val_dice_liver": v_dl_avg, "val_dice_tumor": v_dt_avg,
                "lr": lr, "time_s": epoch_time, "max_grad_norm": max_grad_norm,
                "amp": use_amp, "event": ""})
    pd.DataFrame(log).to_csv("results/training_log.csv", index=False)

    # ★ SAVE LATEST EVERY EPOCH (v6: includes scaler state)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "val_loss": v_loss,
        "best_val_loss": best_val_loss
    }, CKPT_LATEST)
    print(f"  💾 Latest checkpoint saved (epoch {epoch})")

    # Save best model
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        no_improve = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_loss": v_loss,
            "best_val_loss": best_val_loss
        }, CKPT_BEST)
        print(f"  ⭐ BEST MODEL SAVED (val_loss={v_loss:.4f})")
    else:
        no_improve += 1
        print(f"  ⏳ No improvement ({no_improve}/{PATIENCE})")

    # Early stopping
    if no_improve >= PATIENCE:
        print(f"\n🛑 Early stopping at epoch {epoch}")
        break

    # ★ SMART TIMEOUT
    total_elapsed_hrs = (time.time() - global_t0) / 3600.0
    avg_epoch_hrs = np.mean(epoch_times) / 3600.0
    remaining_hrs = MAX_TOTAL_HOURS - total_elapsed_hrs

    if remaining_hrs < avg_epoch_hrs * 1.5:
        print(f"\n⏰ SMART TIMEOUT: {total_elapsed_hrs:.2f}h elapsed, "
              f"avg epoch={avg_epoch_hrs*60:.1f}min, "
              f"remaining={remaining_hrs*60:.1f}min")
        print(f"   Stopping gracefully. Resume by re-running.")
        break

print("\n" + "=" * 60)
print(f"  TRAINING COMPLETE — Best val_loss: {best_val_loss:.4f}")
print(f"  Latest checkpoint: epoch {epoch}")
print(f"  Total time: {(time.time() - global_t0)/3600:.2f} hours")
print("=" * 60)
'''

# ═══════════════════════════════════════════════════════════════
# Apply the patch
# ═══════════════════════════════════════════════════════════════

# Convert the new cell content to notebook format (list of lines)
new_lines = []
for line in NEW_CELL_12.split('\n'):
    new_lines.append(line + '\n')

# Remove trailing empty line with just \n if present
if new_lines and new_lines[-1].strip() == '':
    new_lines[-1] = '\n'

nb['cells'][12]['source'] = new_lines
print(f"✅ Cell 12 replaced ({len(cell12_lines)} lines → {len(new_lines)} lines)")

# ── Verify key fixes are present ──
joined = ''.join(new_lines)
checks = [
    ('GRAD_CLIP    = 5.0', 'Fix #1: GRAD_CLIP raised to 5.0'),
    ('warmup_end_epoch', 'Fix #2: Warmup relative to start_epoch'),
    ('inf_count_this_epoch', 'Fix #3: Inf cascade counter'),
    ('CATASTROPHIC SPIKE', 'Fix #4: Catastrophic loss detection'),
    ('current_epoch - start_ep', 'Fix #5: Relative warmup LR'),
    ('no_improve = 0  # FIX #6', 'Fix #6: Reset patience on rollback'),
    ('v6', 'Version bumped to v6'),
]

all_ok = True
for pattern, desc in checks:
    if pattern in joined:
        print(f"   ✅ {desc}")
    else:
        print(f"   ❌ MISSING: {desc}")
        all_ok = False

if not all_ok:
    print("\n❌ Some fixes are missing! Aborting.")
    sys.exit(1)

# ── Write the patched notebook ──
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅ Notebook saved: {NB_PATH}")
print(f"   Cell 12 now contains v6 Ironclad training pipeline")
print(f"\n📋 CHANGES SUMMARY:")
print(f"   1. GRAD_CLIP: 1.0 → 5.0 (stops crushing healthy gradients)")
print(f"   2. Warmup: always {3} epochs from start_epoch (works on resume)")
print(f"   3. Inf handler: count cascade, auto-disable AMP after {5} events")
print(f"   4. Catastrophic spike: auto-rollback if val_loss > 2× best")
print(f"   5. get_warmup_lr: relative to start_epoch, not absolute")
print(f"   6. Patience reset after rollback (prevents premature early stop)")
