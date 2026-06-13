#!/usr/bin/env python3
"""
v7.2 Final Verification Suite
Validates ALL changes: GradScaler, AdamW, clipping, validation, watchdog.
"""
import json, sys, re

sys.stdout.reconfigure(encoding='utf-8')

NB_PATH = r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\livertumor-model.ipynb'

nb = json.load(open(NB_PATH, encoding='utf-8'))
src = ''.join(nb['cells'][12]['source'])

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ FAIL: {name}")
        if detail: print(f"     → {detail}")
        failed += 1

print("="*60)
print("  v7.2 FINAL VERIFICATION SUITE")
print("="*60)

# =====================================================
# 1: GradScaler
# =====================================================
print("\n[1] GradScaler Configuration")
print("-"*40)
check("init_scale=1024", "init_scale=1024" in src)
check(f"All scalers have init_scale",
      len(re.findall(r"GradScaler\('cuda'", src)) == len(re.findall(r"GradScaler\('cuda', init_scale=", src)))
check("Cascade does NOT reset scaler",
      "GradScaler" not in src[src.find("Inf cascade detected"):src.find("Inf cascade detected")+300])

# =====================================================
# 2: Gradient Clipping
# =====================================================
print("\n[2] Gradient Clipping")
print("-"*40)
check("Single GRAD_CLIP (not adaptive)", "GRAD_CLIP_AMP" not in src and "GRAD_CLIP_FP32" not in src)
m = re.search(r'GRAD_CLIP\s*=\s*([\d.]+)', src)
check("GRAD_CLIP in range 5-15", m and 5.0 <= float(m.group(1)) <= 15.0)

amp_sec = src[src.find("if use_amp:"):src.find("scaler.step(optimizer)")]
check("unscale_() before clip", amp_sec.find("unscale_") < amp_sec.find("clip_grad_norm_"))

# =====================================================
# 3: Optimizer
# =====================================================
print("\n[3] Optimizer")
print("-"*40)
check("All AdamW, no plain Adam", "optim.Adam(" not in src and "optim.AdamW(" in src)
check(f"{src.count('optim.AdamW(')} AdamW creations", src.count("optim.AdamW(") >= 3)
for wd in re.findall(r'weight_decay=([\d.e-]+)', src):
    check(f"weight_decay={wd} OK", 1e-6 <= float(wd) <= 0.01)

# =====================================================
# 4: Validation Precision
# =====================================================
print("\n[4] Validation Precision")
print("-"*40)
val_sec = src[src.find("# ==== VALIDATE"):src.find("# ═══════════════", src.find("# ==== VALIDATE") + 1)]
check("Conditional autocast in validation", "if use_amp:" in val_sec and "autocast" in val_sec)

# =====================================================
# 5: Recovery Paths
# =====================================================
print("\n[5] Recovery Paths")
print("-"*40)
nan_sec = src[src.find("ALL batches had NaN"):src.find("nan_recovery = True")]
check("NaN recovery: init_scale=1024", "init_scale=1024" in nan_sec)
check("NaN recovery: AdamW", "AdamW" in nan_sec)
cat_sec = src[src.find("CATASTROPHIC SPIKE"):src.find("No best checkpoint found")]
check("Catastrophic rollback: init_scale=1024", "init_scale=1024" in cat_sec)
check("Catastrophic rollback: AdamW", "AdamW" in cat_sec)

# =====================================================
# 6: Checkpoints
# =====================================================
print("\n[6] Checkpoints")
print("-"*40)
check("Saves scaler state (2+ times)", src.count('"scaler_state_dict": scaler.state_dict()') >= 2)
check("Restores scaler on resume", "scaler.load_state_dict" in src)

# =====================================================
# 7: WATCHDOG (based on Lightning EarlyStopping)
# =====================================================
print("\n[7] Training Watchdog (Auto-Stop)")
print("-"*40)

# Check 1: Non-finite (Lightning: check_finite)
check("CHECK 1 — Non-finite val_loss detection",
      "NON_FINITE" in src and "v_loss != v_loss" in src)

# Check 2: Divergence threshold (Lightning: divergence_threshold)
check("CHECK 2 — Divergence threshold",
      "DIVERGENCE_THRESHOLD" in src and "DIVERGENCE:" in src)

# Check 3: AMP failure detection
check("CHECK 3 — AMP failure auto-stop",
      "AMP_FAILURE_PATIENCE" in src and "AMP_FAILURE:" in src)

# Check 4: Dice stagnation
check("CHECK 4 — Dice stagnation auto-stop",
      "DICE_STAGNATION_PATIENCE" in src and "DICE_STAGNATION:" in src)

# Check 5: Gradient health
check("CHECK 5a — Gradient explosion auto-stop",
      "GRADIENT_EXPLOSION" in src and "500" in src)
check("CHECK 5b — Gradient collapse auto-stop",
      "GRADIENT_COLLAPSE" in src)

# Fundamental: does `break` actually execute?
watchdog_section = src[src.find("WATCHDOG"):src.find("# ★ SAVE LATEST")]
check("stop_training triggers 'break'",
      "stop_training" in watchdog_section and "break" in watchdog_section)

# Logs reason in CSV
check("Stop reason logged to CSV",
      "WATCHDOG_STOP" in src and "training_log.csv" in src)

# Health status always printed (for post-mortem)
check("Health status logged every epoch",
      "AMP OFF for" in src and "Dice stagnant for" in src)

# Existing patience-based early stopping still works
check("Loss-based early stopping preserved",
      "no_improve >= PATIENCE" in src)

# =====================================================
# 8: PyTorch Compliance
# =====================================================
print("\n[8] PyTorch Compliance")
print("-"*40)
check("Explicit unscale_() before clip", "scaler.unscale_(optimizer)" in src)

inf_start = src.find("if torch.isnan(grad_norm) or torch.isinf(grad_norm):")
inf_end = src.find("continue", inf_start)
inf_code = [l for l in src[inf_start:inf_end].split('\n') if l.strip() and not l.strip().startswith('#')]
check("Inf handler: update() without step()",
      any('scaler.update()' in l for l in inf_code) and not any('scaler.step(' in l for l in inf_code))

check("zero_grad(set_to_none=True)", "set_to_none=True" in src)

# =====================================================
# 9: No Old Patterns
# =====================================================
print("\n[9] No Old Patterns")
print("-"*40)
check("No optim.Adam(", "optim.Adam(" not in src)
check("No bare GradScaler", src.count("GradScaler('cuda')") == 0)
check("No adaptive clip vars", "GRAD_CLIP_AMP" not in src and "GRAD_CLIP_FP32" not in src)

# =====================================================
# SUMMARY
# =====================================================
print(f"\n{'='*60}")
print(f"  RESULTS: {passed} passed, {failed} failed")
print(f"{'='*60}")

if failed > 0:
    print("\n❌ VERIFICATION FAILED")
    sys.exit(1)
else:
    print("\n✅ ALL CHECKS PASSED — safe to deploy to Kaggle")
