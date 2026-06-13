#!/usr/bin/env python3
"""
v7.2 — Add Training Watchdog (auto-stop on failure)

Based on PyTorch Lightning's EarlyStopping pattern:
  1. check_finite   — stop on NaN/Inf val_loss 
  2. divergence     — stop if val_loss > divergence_threshold
  3. patience       — stop if no improvement for N epochs (already exists)
  
Additional stops specific to our AMP pipeline:
  4. amp_failure    — stop if AMP disabled for N consecutive epochs 
  5. dice_stagnation — stop if val Dice hasn't improved for N epochs

Implementation modeled on:
  - PyTorch Lightning EarlyStopping._evaluate_stopping_criteria()
  - Keras TerminateOnNaN callback
  - Standard industry practice for unattended GPU training
"""
import json, sys, os

NB_PATH = r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\livertumor-model.ipynb'

def apply_watchdog():
    nb = json.load(open(NB_PATH, encoding='utf-8'))
    src = ''.join(nb['cells'][12]['source'])
    changes = 0
    
    # ─────────────────────────────────────────────
    # STEP 1: Replace the health monitor warnings with actual STOP logic
    # ─────────────────────────────────────────────
    
    old_health_monitor = """    # ═══════════════════════════════════════════════════════════
    # v7.1: TRAINING HEALTH MONITOR
    # Detects silent failures that waste GPU time
    # ═══════════════════════════════════════════════════════════
    
    # Track AMP status
    if not use_amp:
        consecutive_amp_off += 1
        if consecutive_amp_off >= 3:
            print(f"  🏥 HEALTH: AMP has been OFF for {consecutive_amp_off} consecutive epochs!")
            print(f"     → Training is running 2x slower than necessary")
            print(f"     → Scaler scale: {scaler.get_scale():.1f}")
    else:
        consecutive_amp_off = 0
    
    # Track validation Dice improvement (separate from loss-based early stopping)
    current_val_dice = v_dl_avg  # liver Dice (primary metric for healthcare)
    if current_val_dice > best_val_dice_liver:
        best_val_dice_liver = current_val_dice
        dice_no_improve = 0
    else:
        dice_no_improve += 1
    
    if dice_no_improve >= 7:
        print(f"  🏥 HEALTH: Val Liver Dice has NOT improved for {dice_no_improve} epochs!")
        print(f"     → Current: {current_val_dice:.4f}, Best: {best_val_dice_liver:.4f}")
        print(f"     → Model may be stagnating despite loss changes")
    
    # Gradient health check
    if max_grad_norm > 100.0:
        print(f"  🏥 HEALTH: Gradient norm extremely high ({max_grad_norm:.1f})")
        print(f"     → Expected range: 2-15, current is {max_grad_norm/10:.0f}x too high")
    elif max_grad_norm < 0.01 and n_batches_ok > 10:
        print(f"  🏥 HEALTH: Gradient norm near zero ({max_grad_norm:.6f})")
        print(f"     → Training may have collapsed or learning rate is too low")"""
    
    new_watchdog = """    # ═══════════════════════════════════════════════════════════
    # v7.2: TRAINING WATCHDOG — AUTO-STOP ON FAILURE
    # Based on PyTorch Lightning EarlyStopping pattern:
    #   check_finite → divergence → amp_failure → dice_stagnation → patience
    # ═══════════════════════════════════════════════════════════
    stop_training = False
    stop_reason = ""
    
    # --- CHECK 1: Non-finite val_loss (Lightning: check_finite) ---
    if not (v_loss == float('inf') and n_val == 0):  # skip if no valid batches (handled elsewhere)
        if v_loss != v_loss or v_loss == float('inf') or v_loss == float('-inf'):  # NaN/Inf check
            stop_training = True
            stop_reason = f"NON_FINITE: val_loss={v_loss} is not finite. Training has diverged."
    
    # --- CHECK 2: Loss divergence (Lightning: divergence_threshold) ---
    DIVERGENCE_THRESHOLD = 5.0  # if val_loss exceeds this, model has diverged beyond recovery
    if not stop_training and v_loss > DIVERGENCE_THRESHOLD and epoch > warmup_end_epoch:
        stop_training = True
        stop_reason = f"DIVERGENCE: val_loss={v_loss:.4f} > threshold={DIVERGENCE_THRESHOLD}. Model diverged."
    
    # --- CHECK 3: AMP failure (pipeline-specific) ---
    if not use_amp:
        consecutive_amp_off += 1
    else:
        consecutive_amp_off = 0
    
    AMP_FAILURE_PATIENCE = 5  # if AMP off for 5 consecutive epochs, something is fundamentally wrong
    if not stop_training and consecutive_amp_off >= AMP_FAILURE_PATIENCE:
        stop_training = True
        stop_reason = (f"AMP_FAILURE: AMP has been OFF for {consecutive_amp_off} consecutive epochs. "
                       f"Training is 2x slower and likely unstable. Scaler scale: {scaler.get_scale():.1f}")
    
    # --- CHECK 4: Dice stagnation (metric-specific) ---
    current_val_dice = v_dl_avg  # liver Dice — primary metric for healthcare model
    if current_val_dice > best_val_dice_liver:
        best_val_dice_liver = current_val_dice
        dice_no_improve = 0
    else:
        dice_no_improve += 1
    
    DICE_STAGNATION_PATIENCE = 15  # generous — allows LR reductions to take effect
    if not stop_training and dice_no_improve >= DICE_STAGNATION_PATIENCE and epoch > warmup_end_epoch + 5:
        stop_training = True
        stop_reason = (f"DICE_STAGNATION: Val Liver Dice has NOT improved for {dice_no_improve} epochs. "
                       f"Current: {current_val_dice:.4f}, Best: {best_val_dice_liver:.4f}")
    
    # --- CHECK 5: Gradient health ---
    if not stop_training and max_grad_norm > 500.0:
        stop_training = True
        stop_reason = f"GRADIENT_EXPLOSION: max_grad_norm={max_grad_norm:.1f} > 500. Gradients have exploded."
    elif not stop_training and max_grad_norm < 1e-6 and n_batches_ok > 10 and epoch > warmup_end_epoch:
        stop_training = True
        stop_reason = f"GRADIENT_COLLAPSE: max_grad_norm={max_grad_norm:.8f}. Training has collapsed."
    
    # --- Log health status (always, for post-mortem analysis) ---
    if consecutive_amp_off > 0:
        print(f"  🏥 AMP OFF for {consecutive_amp_off}/{AMP_FAILURE_PATIENCE} epochs")
    if dice_no_improve > 0:
        print(f"  🏥 Dice stagnant for {dice_no_improve}/{DICE_STAGNATION_PATIENCE} epochs (best: {best_val_dice_liver:.4f})")
    if max_grad_norm > 100.0:
        print(f"  🏥 Gradient norm high: {max_grad_norm:.1f} (expected: 2-15)")
    
    # --- EXECUTE STOP ---
    if stop_training:
        print(f"\\n{'🛑'*20}")
        print(f"  WATCHDOG: AUTOMATIC TRAINING STOP")
        print(f"  Reason: {stop_reason}")
        print(f"  Best val_loss: {best_val_loss:.4f}")
        print(f"  Best Liver Dice: {best_val_dice_liver:.4f}")
        print(f"{'🛑'*20}")
        # Save final state before stopping
        log.append({"epoch": epoch, "train_loss": t_loss, "val_loss": v_loss,
                     "train_dice_liver": t_dl_avg, "train_dice_tumor": t_dt_avg,
                     "val_dice_liver": v_dl_avg, "val_dice_tumor": v_dt_avg,
                     "lr": lr, "time_s": epoch_time, "max_grad_norm": max_grad_norm,
                     "amp": use_amp, "event": f"WATCHDOG_STOP: {stop_reason}"})
        pd.DataFrame(log).to_csv("results/training_log.csv", index=False)
        break"""
    
    if old_health_monitor in src:
        src = src.replace(old_health_monitor, new_watchdog)
        changes += 1
        print("✅ Replaced health monitor with auto-stop watchdog")
    else:
        print("❌ Could not find health monitor block")
        return False
    
    # ─────────────────────────────────────────────
    # STEP 2: Update the training complete message to show stop reason
    # ─────────────────────────────────────────────
    old_complete = """print("\\n" + "=" * 60)
print(f"  TRAINING COMPLETE — Best val_loss: {best_val_loss:.4f}")
print(f"  Latest checkpoint: epoch {epoch}")
print(f"  Total time: {(time.time() - global_t0)/3600:.2f} hours")
print("=" * 60)"""
    
    new_complete = """print("\\n" + "=" * 60)
print(f"  TRAINING COMPLETE — Best val_loss: {best_val_loss:.4f}")
print(f"  Best Liver Dice: {best_val_dice_liver:.4f}")
print(f"  Latest checkpoint: epoch {epoch}")
print(f"  Total time: {(time.time() - global_t0)/3600:.2f} hours")
if stop_training:
    print(f"  Stop reason: {stop_reason}")
elif no_improve >= PATIENCE:
    print(f"  Stop reason: EARLY_STOPPING (patience={PATIENCE} exhausted)")
else:
    print(f"  Stop reason: COMPLETED all {EPOCHS} epochs")
print("=" * 60)"""
    
    if old_complete in src:
        src = src.replace(old_complete, new_complete)
        changes += 1
        print("✅ Updated training complete message")
    else:
        print("❌ Could not find training complete block")
        return False
    
    # ─────────────────────────────────────────────
    # WRITE BACK
    # ─────────────────────────────────────────────
    cell_lines = src.split('\n')
    new_source = []
    for i, line in enumerate(cell_lines):
        if i < len(cell_lines) - 1:
            new_source.append(line + '\n')
        else:
            new_source.append(line)
    
    nb['cells'][12]['source'] = new_source
    
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print(f"\n{'='*60}")
    print(f"  ✅ WATCHDOG APPLIED ({changes} changes)")
    print(f"{'='*60}")
    return True

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    success = apply_watchdog()
    if not success:
        print("\n❌ PATCH FAILED")
        sys.exit(1)
