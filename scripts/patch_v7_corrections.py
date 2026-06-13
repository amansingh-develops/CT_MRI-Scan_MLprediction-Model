#!/usr/bin/env python3
"""
v7.1 Corrections — Fix issues found during verification:
  1. REMOVE adaptive gradient clipping (PyTorch docs: after unscale_(), norms are already FP32-equivalent)
  2. ADD training health monitoring (detect stagnation, AMP failures, gradient anomalies)
"""
import json, sys, os, shutil

NB_PATH = r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\livertumor-model.ipynb'

def apply_corrections():
    nb = json.load(open(NB_PATH, encoding='utf-8'))
    src = ''.join(nb['cells'][12]['source'])
    changes = 0
    
    # ─────────────────────────────────────────────
    # FIX 1: Single GRAD_CLIP instead of adaptive
    # Per PyTorch docs: after scaler.unscale_(), gradient norms are restored
    # to FP32-equivalent magnitude. Same max_norm works for both.
    # Using 10.0 (not 5.0) as safety margin since model norms are 2-12.
    # ─────────────────────────────────────────────
    old_clip_config = """GRAD_CLIP_AMP  = 5.0         # v7: AMP gradient norms are 2-12
GRAD_CLIP_FP32 = 50.0        # v7: FP32 gradient norms are 17-100"""
    new_clip_config = """GRAD_CLIP      = 10.0        # v7.1: single clip for both AMP & FP32 (norms are 2-12 after unscale_)"""
    
    if old_clip_config in src:
        src = src.replace(old_clip_config, new_clip_config)
        changes += 1
        print("✅ Fix 1a: Replaced adaptive clip with single GRAD_CLIP=10.0")
    else:
        print("❌ Could not find adaptive clip config")
        return False
    
    # Replace all clip_val usages back to GRAD_CLIP
    old_clip_usage = """clip_val = GRAD_CLIP_AMP if use_amp else GRAD_CLIP_FP32
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)"""
    new_clip_usage = """grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)"""
    
    count = src.count(old_clip_usage)
    if count >= 1:
        src = src.replace(old_clip_usage, new_clip_usage)
        changes += 1
        print(f"✅ Fix 1b: Replaced {count} adaptive clip usages with GRAD_CLIP")
    else:
        print("❌ Could not find adaptive clip usage")
        return False
    
    # Update header
    src = src.replace(
        'print(f"Grad clip:    AMP={GRAD_CLIP_AMP} / FP32={GRAD_CLIP_FP32} (adaptive)")',
        'print(f"Grad clip:    {GRAD_CLIP} (after unscale_, same for AMP & FP32)")'
    )
    changes += 1
    print("✅ Fix 1c: Updated header print")
    
    # ─────────────────────────────────────────────
    # FIX 2: Add Training Health Monitor
    # Detects: AMP stuck off, gradient anomalies, dice stagnation
    # ─────────────────────────────────────────────
    
    # Add health tracking variables after the existing state vars
    old_state_vars = """nan_recovery  = False    # v6: True if we just recovered from NaN
use_amp       = True     # v6: can be disabled during recovery
warmup_end_epoch = start_epoch + WARMUP_EPOCHS - 1  # v6: warmup ends here"""
    
    new_state_vars = """nan_recovery  = False    # v6: True if we just recovered from NaN
use_amp       = True     # v6: can be disabled during recovery
warmup_end_epoch = start_epoch + WARMUP_EPOCHS - 1  # v6: warmup ends here

# v7.1: Training Health Monitor
consecutive_amp_off = 0     # count epochs where AMP stayed off
best_val_dice_liver = 0.0   # track best validation Dice (separate from loss)
dice_no_improve = 0         # epochs where val Dice didn't improve"""
    
    if old_state_vars in src:
        src = src.replace(old_state_vars, new_state_vars)
        changes += 1
        print("✅ Fix 2a: Added health monitoring state variables")
    else:
        print("❌ Could not find state vars block")
        return False
    
    # Add health check after epoch logging (before save checkpoint)
    old_after_log = """    pd.DataFrame(log).to_csv("results/training_log.csv", index=False)

    # ★ SAVE LATEST EVERY EPOCH (v6: includes scaler state)"""
    
    new_after_log = """    pd.DataFrame(log).to_csv("results/training_log.csv", index=False)

    # ═══════════════════════════════════════════════════════════
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
        print(f"     → Training may have collapsed or learning rate is too low")

    # ★ SAVE LATEST EVERY EPOCH (v6: includes scaler state)"""
    
    if old_after_log in src:
        src = src.replace(old_after_log, new_after_log)
        changes += 1
        print("✅ Fix 2b: Added per-epoch health monitoring")
    else:
        print("❌ Could not find logging block to insert health monitor")
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
    print(f"  ✅ ALL {changes} CORRECTIONS APPLIED")
    print(f"{'='*60}")
    return True

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    success = apply_corrections()
    if not success:
        print("\n❌ CORRECTIONS FAILED — notebook unchanged")
        sys.exit(1)
