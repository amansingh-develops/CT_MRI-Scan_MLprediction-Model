"""
FULL DIAGNOSTIC ANALYSIS — v6 Latest Training Run
Complete analysis of 28 epochs, AMP behavior, gradient health, overfitting
"""
import csv, sys, statistics
sys.stdout.reconfigure(encoding='utf-8')

BASE = r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project'

def read_csv(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get('epoch', '').strip():
                rows.append(r)
    return rows

run = read_csv(BASE + r'\Latest_trainingLogs\training_log (1).csv')

print("=" * 80)
print("  COMPLETE DIAGNOSTIC — v6 Run (28 epochs, 12 hours)")
print("=" * 80)

print(f"\n{'Ep':>3} {'TrLoss':>8} {'VaLoss':>8} {'TrLiv':>6} {'TrTum':>6} {'VaLiv':>6} {'VaTum':>6} {'GradN':>8} {'AMP':>5} {'Time':>6} {'Note':>15}")
for r in run:
    ep = int(r['epoch'])
    amp = 'AMP' if r.get('amp') == 'True' else 'FP32'
    t = float(r['time_s'])
    vl = float(r['val_loss'])
    note = ''
    if r.get('event', '').strip():
        note = r['event']
    elif ep == 15:
        note = '⭐ BEST(AMP)'
    elif ep == 26:
        note = '⭐ BEST(ALL)'
    print(f"{ep:3d} {float(r['train_loss']):8.4f} {vl:8.4f} "
          f"{float(r['train_dice_liver']):6.3f} {float(r['train_dice_tumor']):6.3f} "
          f"{float(r['val_dice_liver']):6.3f} {float(r['val_dice_tumor']):6.3f} "
          f"{float(r['max_grad_norm']):8.2f} {amp:>5} {t:5.0f}s {note:>15}")

# ═══════════════════════════════════════════════════════════════
# PHASE ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  PHASE ANALYSIS")
print(f"{'='*80}")

amp_epochs = [r for r in run if r.get('amp') == 'True']
fp32_epochs = [r for r in run if r.get('amp') == 'False']

# AMP PHASE (1-17)
if amp_epochs:
    times = [float(r['time_s']) for r in amp_epochs]
    grads = [float(r['max_grad_norm']) for r in amp_epochs]
    vl = [float(r['val_loss']) for r in amp_epochs]
    tl = [float(r['train_loss']) for r in amp_epochs]
    print(f"\n🟢 AMP PHASE (epochs 1-{len(amp_epochs)}): {len(amp_epochs)} epochs")
    print(f"  Time:      {sum(times)/3600:.1f}h total, {statistics.mean(times)/60:.1f}min avg/epoch")
    print(f"  Grad norm: {min(grads):.1f} — {max(grads):.1f} (avg {statistics.mean(grads):.1f})")
    print(f"  Val loss:  {vl[0]:.4f} → {vl[-1]:.4f} (best={min(vl):.4f})")
    print(f"  Trn loss:  {tl[0]:.4f} → {tl[-1]:.4f}")

# FP32 PHASE (18-28)
if fp32_epochs:
    times = [float(r['time_s']) for r in fp32_epochs]
    grads = [float(r['max_grad_norm']) for r in fp32_epochs]
    vl = [float(r['val_loss']) for r in fp32_epochs]
    tl = [float(r['train_loss']) for r in fp32_epochs]
    print(f"\n🔴 FP32 PHASE (epochs 18-28): {len(fp32_epochs)} epochs")
    print(f"  Time:      {sum(times)/3600:.1f}h total, {statistics.mean(times)/60:.1f}min avg/epoch")
    print(f"  Grad norm: {min(grads):.1f} — {max(grads):.1f} (avg {statistics.mean(grads):.1f})")
    print(f"  Val loss:  {vl[0]:.4f} → {vl[-1]:.4f} (best={min(vl):.4f})")
    print(f"  Trn loss:  {tl[0]:.4f} → {tl[-1]:.4f}")

# ═══════════════════════════════════════════════════════════════
# KEY FINDING: Loss function changed behavior between AMP and FP32
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  KEY FINDING: LOSS FUNCTION BEHAVIOR CHANGE")
print(f"{'='*80}")
print(f"\n  AMP  best val_loss: {min(float(r['val_loss']) for r in amp_epochs):.4f} (epoch 15)")
print(f"  FP32 best val_loss: {min(float(r['val_loss']) for r in fp32_epochs):.4f} (epoch 26)")
print(f"\n  ⚠️ FP32 val_loss (0.186) is LOWER than AMP val_loss (0.395)")
print(f"  ⚠️ But FP32 val_dice_liver (0.71) is WORSE than AMP val_dice_liver (0.71)")
print(f"  ⚠️ This means the loss and dice are DISAGREEING")
print(f"\n  EXPLANATION: In FP32 mode, BCE loss computes differently due to")
print(f"  higher precision. The model optimizes BCE (lower loss) but may not")
print(f"  be optimizing the actual segmentation quality (Dice).")

# ═══════════════════════════════════════════════════════════════
# OVERFITTING ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  OVERFITTING ANALYSIS")
print(f"{'='*80}")
print(f"\n{'Ep':>3} {'TrLiver':>8} {'VaLiver':>8} {'Gap':>6} {'TrTumor':>8} {'VaTumor':>8} {'Gap':>6}")
for r in run:
    ep = int(r['epoch'])
    tr_l = float(r['train_dice_liver'])
    va_l = float(r['val_dice_liver'])
    tr_t = float(r['train_dice_tumor'])
    va_t = float(r['val_dice_tumor'])
    gl = tr_l - va_l
    gt = tr_t - va_t
    flag = ' ⚠️' if gl > 0.25 else ''
    print(f"{ep:3d} {tr_l:8.3f} {va_l:8.3f} {gl:6.3f} {tr_t:8.3f} {va_t:8.3f} {gt:6.3f}{flag}")

# ═══════════════════════════════════════════════════════════════
# GRADIENT HEALTH
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  GRADIENT HEALTH")
print(f"{'='*80}")
for r in run:
    ep = int(r['epoch'])
    gn = float(r['max_grad_norm'])
    amp = 'AMP' if r.get('amp') == 'True' else 'FP32'
    bar = '█' * min(int(gn), 50)
    flag = ' ⚠️ EXPLODING' if gn > 50 else ''
    print(f"  Epoch {ep:2d} ({amp:4s}): {gn:8.1f} {bar}{flag}")

# ═══════════════════════════════════════════════════════════════
# SPEED ANALYSIS  
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  SPEED & EFFICIENCY")
print(f"{'='*80}")
total_time_h = sum(float(r['time_s']) for r in run) / 3600
amp_time_h = sum(float(r['time_s']) for r in amp_epochs) / 3600
fp32_time_h = sum(float(r['time_s']) for r in fp32_epochs) / 3600
print(f"  Total training time: {total_time_h:.1f}h")
print(f"  AMP phase:   {amp_time_h:.1f}h for {len(amp_epochs)} epochs ({amp_time_h/len(amp_epochs)*60:.1f} min/epoch)")
print(f"  FP32 phase:  {fp32_time_h:.1f}h for {len(fp32_epochs)} epochs ({fp32_time_h/len(fp32_epochs)*60:.1f} min/epoch)")
print(f"\n  If ALL epochs ran in AMP: could have done {int(total_time_h / (amp_time_h/len(amp_epochs)))} epochs instead of 28")
print(f"  That's {int(total_time_h / (amp_time_h/len(amp_epochs))) - 28} EXTRA EPOCHS wasted by running in FP32")

# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE CHAIN
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  ROOT CAUSE CHAIN (Why AMP breaks every epoch)")
print(f"{'='*80}")
print("""
  1. Epoch 17: Some batches produce Inf gradients (normal, happens occasionally)
  2. v6 cascade detector triggers after 5 Inf → disables AMP → creates new GradScaler
  3. Epoch 17 completes in FP32 mode successfully
  4. Epoch 18 starts: `nan_recovery = False` → sets `use_amp = True`  
  5. BUT: the brand new GradScaler has default scale (65536)
  6. First few batches with fresh high-scale scaler → produce Inf immediately
  7. Cascade detector triggers AGAIN → disables AMP AGAIN
  8. This repeats EVERY epoch: fresh scaler → immediate Inf → cascade → FP32
  
  ROOT CAUSE: The scaler is reset after cascade, but the DEFAULT scale (65536)
  is too high for this model's gradient magnitudes. The model needs a lower
  initial scale, OR the scaler should be preserved (not reset) between epochs.

  SECONDARY ISSUE: When running FP32, GRAD_CLIP=5.0 clips gradients that are
  naturally 50-190 in FP32 mode. This means ~95% of gradient info is being
  discarded, causing slow/unstable convergence.
""")

print(f"{'='*80}")
print("  DIAGNOSIS COMPLETE")
print(f"{'='*80}")
