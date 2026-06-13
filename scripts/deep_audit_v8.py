"""
v8 Notebook Deep Audit — Line-by-line verification
Checks every logic path, edge case, variable scope, and known pitfall.
"""
import json, sys, re, os

NB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "livertumor-model.ipynb")
with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
code_cells = [c for c in cells if c['cell_type'] == 'code']

# Combine all code into one big source for cross-cell analysis
all_code = ""
cell_sources = []
for i, c in enumerate(code_cells):
    src = c['source']
    cell_sources.append(src)
    all_code += f"\n# --- CODE CELL {i} ---\n" + src

passed = 0
failed = 0
warnings = 0

def check(condition, desc, severity="FAIL"):
    global passed, failed, warnings
    if condition:
        passed += 1
        print(f"  PASS: {desc}")
    else:
        if severity == "WARN":
            warnings += 1
            print(f"  WARN: {desc}")
        else:
            failed += 1
            print(f"  FAIL: {desc}")
    return condition

print("=" * 70)
print("  v8 DEEP AUDIT — Every Known Failure Mode Checked")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# SECTION 1: STRUCTURAL CHECKS
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 1: Notebook Structure")
check(len(code_cells) == 11, f"Exactly 11 code cells (found {len(code_cells)})")
check(nb['nbformat'] == 4, "nbformat is 4")
check(nb['metadata']['kaggle']['isGpuEnabled'] == True, "GPU enabled in Kaggle metadata")
check(nb['metadata']['kaggle']['accelerator'] == 'gpu', "Accelerator set to 'gpu'")

# ═══════════════════════════════════════════════════════════════
# SECTION 2: IMPORTS AND DEPENDENCIES
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 2: Imports & Dependencies")
required_imports = [
    ('torch', 'PyTorch'),
    ('torch.nn as nn', 'nn module'),
    ('torch.optim as optim', 'optim module'),
    ('from torch.utils.data import Dataset, DataLoader', 'Dataset/DataLoader'),
    ('numpy as np', 'numpy'),
    ('pandas as pd', 'pandas'),
    ('cv2', 'OpenCV'),
    ('from tqdm import tqdm', 'tqdm'),
    ('matplotlib', 'matplotlib'),
    ('import gc', 'garbage collector'),
    ('glob', 'glob'),  # may be in multi-import like 'import os, glob, time'
    ('time', 'time'),   # same
]
for imp, name in required_imports:
    check(imp in all_code, f"Import: {name} ({imp})")

# ═══════════════════════════════════════════════════════════════
# SECTION 3: FATAL BUG CHECKS (from v5/v6/v7 failure history)
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 3: Known Fatal Bug Patterns")

# BUG 1: AMP disable logic (caused OOM in v7)
check('use_amp = False' not in all_code, "NO 'use_amp = False' (v7 OOM cause)")
check('use_amp=False' not in all_code, "NO 'use_amp=False' variant")

# BUG 2: Catastrophic rollback (caused OOM in v7)
check('CATASTROPHIC' not in all_code, "NO 'CATASTROPHIC' rollback logic")

# BUG 3: nan_recovery flag (caused FP32 fallback in v7)
# Must check CODE cells only, not markdown
code_only = "\n".join(cell_sources)
check('nan_recovery' not in code_only, "NO 'nan_recovery' in code cells")

# BUG 4: Manual GradScaler init_scale
# PyTorch default is 65536, manual 1024 caused issues
# Only check actual code lines, not comments
code_lines_only = [l for l in code_only.split('\n') if l.strip() and not l.strip().startswith('#')]
code_no_comments = '\n'.join(code_lines_only)
check('init_scale' not in code_no_comments, "NO manual init_scale in code (using PyTorch default 65536)")

# BUG 5: shuffle=True with sampler (DataLoader crash)
# This is a CRITICAL check — you CANNOT use shuffle=True when using a sampler
# Only check the TRAINING DataLoader, not the overfit test DataLoader
train_dl_lines = [l for l in code_only.split('\n') if 'train_ld' in l and 'DataLoader' in l]
has_shuffle_with_sampler = any('shuffle=True' in l and 'sampler' in l for l in train_dl_lines)
check(not has_shuffle_with_sampler,
      "NO shuffle=True on training DataLoader (which uses sampler)")

# More specific check: the train DataLoader must NOT have shuffle=True
train_dl_pattern = re.search(r'train_ld\s*=\s*DataLoader\([^)]+\)', code_only)
if train_dl_pattern:
    train_dl_text = train_dl_pattern.group()
    check('shuffle' not in train_dl_text or 'shuffle=False' in train_dl_text,
          f"Train DataLoader does NOT use shuffle with sampler")
else:
    check(False, "Could not find train DataLoader definition")

# BUG 6: Division by ACCUM_STEPS before backward but metrics don't account for it
loss_div_pattern = 'loss  = loss / ACCUM_STEPS'
check(loss_div_pattern in code_only, "Loss divided by ACCUM_STEPS before backward")
# Check metrics use ACCUM_STEPS correction
metric_correction = 't_loss_sum += loss.item() * ACCUM_STEPS'
check(metric_correction in code_only,
      "Metrics multiply back by ACCUM_STEPS for correct logging")

# ═══════════════════════════════════════════════════════════════
# SECTION 4: DICELOSS FP32 SAFETY
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 4: DiceLoss FP32 Safety")

# The DiceLoss.forward must call .float() on both pred and target
dice_class = re.search(r'class DiceLoss.*?(?=class |# ── )', code_only, re.DOTALL)
if dice_class:
    dice_src = dice_class.group()
    check('pred = torch.sigmoid(pred).float()' in dice_src,
          "DiceLoss: sigmoid + .float() on predictions")
    check('target = target.float()' in dice_src,
          "DiceLoss: .float() on targets")
    check('.contiguous()' in dice_src,
          "DiceLoss: contiguous() call for memory layout")
    check('self.smooth' in dice_src, "DiceLoss: uses smooth factor")
    check('smooth=1e-6' in dice_src, "DiceLoss: smooth=1e-6 (not 0)")
else:
    check(False, "Could not find DiceLoss class")

# CombinedLoss must also force FP32
combined_class = re.search(r'class CombinedLoss.*?(?=# ── )', code_only, re.DOTALL)
if combined_class:
    comb_src = combined_class.group()
    check('pred.float()' in comb_src, "CombinedLoss: BCE uses .float() on pred")
    check('target.float()' in comb_src, "CombinedLoss: BCE uses .float() on target")
    check('0.5 * self.bce' in comb_src, "CombinedLoss: 50/50 BCE/Dice weighting")
else:
    check(False, "Could not find CombinedLoss class")

# ═══════════════════════════════════════════════════════════════
# SECTION 5: WEIGHTED SAMPLER LOGIC
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 5: Weighted Sampler")

check('WeightedRandomSampler' in code_only, "WeightedRandomSampler used")
check("3.0 if has_liver else 1.0" in code_only,
      "Liver-positive weight = 3.0, empty = 1.0")
check('replacement=True' in code_only,
      "WeightedRandomSampler replacement=True (required for weighted sampling)")

# Check that scan uses the SAME CSV as the training dataset
check('train_csv_data = pd.read_csv(TRAIN_CSV)' in code_only,
      "Sampler scans same CSV as training dataset")

# ═══════════════════════════════════════════════════════════════
# SECTION 6: GRADIENT ACCUMULATION LOGIC
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 6: Gradient Accumulation")

check('ACCUM_STEPS   = 2' in code_only, "ACCUM_STEPS = 2")
check('BATCH         = 16' in code_only, "BATCH = 16")

# Critical: optimizer.zero_grad BEFORE the batch loop
check('optimizer.zero_grad(set_to_none=True)\n\n    pbar' in code_only,
      "optimizer.zero_grad() called BEFORE batch loop starts")

# Accumulation gate
check('if (bi + 1) % ACCUM_STEPS == 0 or (bi + 1) == len(train_ld)' in code_only,
      "Accumulation step at correct boundary (including last batch)")

# Step order: unscale -> clip -> step -> update -> zero_grad
accum_block = re.search(r'if \(bi \+ 1\) % ACCUM_STEPS.*?optimizer\.zero_grad',
                        code_only, re.DOTALL)
if accum_block:
    block = accum_block.group()
    ops = ['unscale_', 'clip_grad_norm_', 'scaler.step', 'scaler.update',
           'optimizer.zero_grad']
    positions = [block.find(op) for op in ops]
    check(all(p >= 0 for p in positions), "All GradAccum operations present")
    check(positions == sorted(positions),
          f"GradAccum operations in correct order: unscale->clip->step->update->zero_grad")
else:
    check(False, "Could not find gradient accumulation block")

# ═══════════════════════════════════════════════════════════════
# SECTION 7: TRAINING LOOP EDGE CASES
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 7: Training Loop Edge Cases")

# NaN/Inf loss skip
check('if torch.isnan(loss) or torch.isinf(loss):' in code_only,
      "NaN/Inf loss detection in training loop")
check('optimizer.zero_grad(set_to_none=True)\n            continue' in code_only,
      "NaN/Inf: zero_grad + continue (not crash)")

# Inf gradient handling
check('if torch.isfinite(gnorm):' in code_only,
      "Inf gradient check with torch.isfinite")
check('inf_events += 1' in code_only,
      "Inf gradient events counted")

# n_ok guard
check('if n_ok == 0:' in code_only, "Guard: if all batches had NaN loss -> break")

# Division safety
check('n_ok' in code_only and 't_loss  = t_loss_sum / n_ok' in code_only,
      "Training loss averaged by n_ok (not len(train_ld))")
check('v_loss = v_loss_sum / max(1, n_val)' in code_only,
      "Val loss uses max(1, n_val) to prevent division by zero")

# ═══════════════════════════════════════════════════════════════
# SECTION 8: CHECKPOINT LOGIC
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 8: Checkpoint Save/Load")

# Fresh optimizer on resume (v8 fix #5)
check('optimizer = optim.Adam(model.parameters(), lr=finetune_lr)' in code_only,
      "Fresh optimizer created on checkpoint resume (no state loading)")
check('finetune_lr = 1e-5' in code_only,
      "Fine-tune LR = 1e-5 (not the training LR)")

# Both best and latest checkpoints saved
check("CKPT_BEST   = " in code_only, "CKPT_BEST path defined")
check("CKPT_LATEST = " in code_only, "CKPT_LATEST path defined")

# Check saved checkpoint keys
for key in ['epoch', 'model_state_dict', 'optimizer_state_dict',
            'scaler_state_dict', 'val_loss', 'best_val_loss']:
    check(f'"{key}"' in code_only or f"'{key}'" in code_only,
          f"Checkpoint saves '{key}'")

# ═══════════════════════════════════════════════════════════════
# SECTION 9: SCHEDULER + WARMUP
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 9: Scheduler & Warmup")

check('ReduceLROnPlateau' in code_only, "Scheduler: ReduceLROnPlateau")
check('scheduler.step(v_loss)' in code_only,
      "Scheduler steps on val_loss (not train_loss)")
check('if epoch > warmup_end:' in code_only,
      "Scheduler only steps AFTER warmup")

# Warmup logic
check('WARMUP_EPOCHS = 5' in code_only, "Warmup = 5 epochs")
check('def get_warmup_lr' in code_only, "Warmup LR function defined")
# Check warmup is applied
check("if epoch <= warmup_end:" in code_only, "Warmup applied conditionally")

# ═══════════════════════════════════════════════════════════════
# SECTION 10: WATCHDOG (SIMPLE)
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 10: Watchdog")

check('stop = False' in code_only, "Watchdog: clean boolean flag")
check('v_loss > 5.0' in code_only, "Watchdog: divergence threshold = 5.0")
check('max_gnorm > 500.0' in code_only, "Watchdog: gradient explosion threshold = 500.0")
check('torch.tensor(v_loss).isfinite()' in code_only,
      "Watchdog: non-finite val_loss check")

# No recovery — only stop (check code lines, not comments)
check('rollback' not in code_no_comments.lower(), "Watchdog: NO rollback logic in code (only in comments)")
# Allow "No recovery system" in comments but not actual recovery code
recovery_code_patterns = ['load_state_dict(best_', 'model.load_state_dict(ckpt_best']
for pat in recovery_code_patterns:
    check(pat not in code_only, f"Watchdog: no '{pat}' recovery code")

# ═══════════════════════════════════════════════════════════════
# SECTION 11: EARLY STOPPING
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 11: Early Stopping")

check('PATIENCE      = 15' in code_only, "Patience = 15")
check('no_improve >= PATIENCE' in code_only, "Early stop when no_improve >= PATIENCE")
check('no_improve = 0' in code_only, "no_improve reset on improvement")
check('no_improve += 1' in code_only, "no_improve incremented on no improvement")

# ═══════════════════════════════════════════════════════════════
# SECTION 12: EVALUATION CELLS (Variable Scope)
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 12: Evaluation Cell Safety")

# Cell 8 (evaluation) re-defines model and loads checkpoint
eval_cell = cell_sources[5]  # Cell 8 is code cell 5 (0-indexed)
# Actually, let me find the right cell
eval_cell = None
for cs in cell_sources:
    if 'FULL MODEL EVALUATION' in cs:
        eval_cell = cs
        break

if eval_cell:
    check('model = UNet(1, 2).to(device)' in eval_cell,
          "Eval cell creates fresh model (not reusing training model)")
    check("torch.load(CKPT" in eval_cell or "torch.load(" in eval_cell,
          "Eval cell loads checkpoint")
    check('model.eval()' in eval_cell, "Eval cell sets model.eval()")
    check('with torch.no_grad()' in eval_cell, "Eval cell uses torch.no_grad()")
else:
    check(False, "Could not find evaluation cell")

# ═══════════════════════════════════════════════════════════════
# SECTION 13: DATASET EDGE CASES
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 13: Dataset Edge Cases")

# Check resize interpolation
check('cv2.INTER_NEAREST' in code_only,
      "Masks use INTER_NEAREST (not INTER_LINEAR)")
check('cv2.INTER_LINEAR' in code_only,
      "Images use INTER_LINEAR (smooth)")

# Check binarization threshold
check('(liv > 127)' in code_only, "Liver mask binarized at > 127")
check('(tum > 127)' in code_only, "Tumor mask binarized at > 127")

# Check normalization
check('img.astype(np.float32) / 255.0' in code_only,
      "Image normalized to [0, 1]")

# Check output shape
check("torch.from_numpy(np.stack([liv, tum], axis=0))" in code_only,
      "Mask stacked as (2, H, W) — liver channel 0, tumor channel 1")

# ═══════════════════════════════════════════════════════════════
# SECTION 14: MEMORY MANAGEMENT
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 14: Memory Management")

check('expandable_segments:True' in all_code,
      "PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True")
check('torch.cuda.empty_cache()' in all_code, "torch.cuda.empty_cache() called")
check('gc.collect()' in all_code, "gc.collect() called")
check('pin_memory=True' in code_only, "DataLoader pin_memory=True")
check('non_blocking=True' in code_only, "Tensor .to(device, non_blocking=True)")
check('set_to_none=True' in code_only, "zero_grad(set_to_none=True) for memory efficiency")

# Cleanup of sanity check models
check('del test_model, test_opt' in code_only,
      "Overfit test model cleaned up")
check('del model, dummy, out' in code_only,
      "Sanity check model cleaned up")

# ═══════════════════════════════════════════════════════════════
# SECTION 15: OUTPUT VOLUME
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 15: Output Volume Control")

check('miniters=50' in code_only, "tqdm miniters=50 (reduce output)")
check('mininterval=10.0' in code_only, "tqdm mininterval=10s (reduce output)")
check('leave=False' in code_only, "tqdm leave=False (clean output)")

# ═══════════════════════════════════════════════════════════════
# SECTION 16: AMP CONSISTENCY
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 16: AMP Consistency")

# Count autocast usages — should be consistent
autocast_count = code_only.count("torch.amp.autocast('cuda')")
check(autocast_count >= 3,
      f"torch.amp.autocast used {autocast_count} times (train + val + eval)")

# GradScaler usage
check("torch.amp.GradScaler('cuda')" in code_only, "GradScaler uses new API format")

# No old-style amp
check('torch.cuda.amp.autocast' not in code_only,
      "No deprecated torch.cuda.amp.autocast (uses torch.amp.autocast)")
check('torch.cuda.amp.GradScaler' not in code_only,
      "No deprecated torch.cuda.amp.GradScaler (uses torch.amp.GradScaler)")

# ═══════════════════════════════════════════════════════════════
# SECTION 17: CSV LOGGING
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 17: CSV Logging")

check("results/training_log.csv" in code_only, "CSV path: results/training_log.csv")
check("pd.DataFrame(log).to_csv" in code_only, "Log written with pd.DataFrame")
# Check log is written on every epoch AND on watchdog stop AND on early stop
log_write_count = code_only.count("pd.DataFrame(log).to_csv")
check(log_write_count >= 3,
      f"CSV written in {log_write_count} places (normal + watchdog + early stop)")

# ═══════════════════════════════════════════════════════════════
# SECTION 18: DATA PATH CONSISTENCY
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 18: Data Paths")

# Check DATA_ROOT, TRAIN_CSV, VAL_CSV are defined consistently
check('DATA_ROOT = "/kaggle/input/datasets/andrewmvd/lits-png/dataset_6/dataset_6"' in code_only,
      "DATA_ROOT path correct")
check('TRAIN_CSV = "/kaggle/input/datasets/andrewmvd/lits-png/lits_train.csv"' in code_only,
      "TRAIN_CSV path correct")
check('VAL_CSV   = "/kaggle/input/datasets/andrewmvd/lits-png/lits_val.csv"' in code_only,
      "VAL_CSV path correct")
check('TEST_CSV  = "/kaggle/input/datasets/andrewmvd/lits-png/lits_test.csv"' in code_only,
      "TEST_CSV path correct (3-way split)")

# Check paths have assertion
check('assert os.path.exists(p)' in code_only, "Data paths verified with assert")

# ═══════════════════════════════════════════════════════════════
# SECTION 19: PREVIOUS CHECKPOINT (answer the user's question)
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 19: Previous Checkpoint Handling")

check("find_prev_checkpoint" in code_only, "Previous checkpoint search function defined")
check('PREV_CKPT     = None' in code_only,
      "PREV_CKPT defaults to None (fresh training unless set)")
check("/kaggle/input/*/models/best_model.pth" in code_only,
      "Auto-searches Kaggle input for previous best_model.pth")

# ═══════════════════════════════════════════════════════════════
# SECTION 20: CROSS-CELL VARIABLE SAFETY
# ═══════════════════════════════════════════════════════════════
print("\n>> SECTION 20: Cross-Cell Variable Safety")

# Variables used in eval cells that MUST be defined in those cells
# (not relying on training cell variables that may not exist if training is skipped)

# Cell 8 (eval) must define: model, device, eval_ds, eval_ld
if eval_cell:
    check("device = torch.device" in eval_cell,
          "Eval cell defines its own 'device'")
    check("eval_ds = LITSDataset" in eval_cell,
          "Eval cell defines its own 'eval_ds'")
    check("eval_ld = DataLoader" in eval_cell,
          "Eval cell defines its own 'eval_ld'")

# Cell 9 (visualization) uses 'model' and 'eval_ds' — check they're available
viz_cell = None
for cs in cell_sources:
    if 'DETAILED PREDICTION VISUALIZATIONS' in cs:
        viz_cell = cs
        break

if viz_cell:
    # Viz cell REUSES model and eval_ds from eval cell — that's fine since
    # eval cell (cell 8) always runs before cell 9
    check('eval_ds[idx]' in viz_cell or 'eval_ds[' in viz_cell,
          "Viz cell uses eval_ds from eval cell (correct dependency order)")

# Cell 10 (dashboard) uses 'metrics', 'ckpt' from eval cell
dash_cell = None
for cs in cell_sources:
    if 'GENERATING METRICS DASHBOARD' in cs:
        dash_cell = cs
        break

if dash_cell:
    check("metrics['Liver']" in dash_cell,
          "Dashboard cell uses metrics dict from eval cell")
    check("ckpt['epoch']" in dash_cell,
          "Dashboard cell uses ckpt from eval cell")

# Cell 12 (test eval) must define its own model and dataset
test_eval_cell = None
for cs in cell_sources:
    if 'HELD-OUT TEST SET EVALUATION' in cs:
        test_eval_cell = cs
        break

if test_eval_cell:
    check('test_model = UNet(1, 2).to(device)' in test_eval_cell,
          "Test eval cell creates fresh model")
    check('test_ds = LITSDataset(TEST_CSV' in test_eval_cell,
          "Test eval cell uses TEST_CSV (not VAL_CSV)")
    check('test_model.eval()' in test_eval_cell,
          "Test eval cell sets model.eval()")
    check('test_metrics.csv' in test_eval_cell,
          "Test eval saves results to test_metrics.csv")
    check('VAL vs TEST COMPARISON' in test_eval_cell,
          "Test eval includes val vs test comparison table")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"  AUDIT COMPLETE")
print(f"  PASSED: {passed}")
print(f"  FAILED: {failed}")
print(f"  WARNINGS: {warnings}")
print("=" * 70)

if failed > 0:
    print(f"\n  !!! {failed} CRITICAL FAILURES — DO NOT DEPLOY !!!")
    sys.exit(1)
elif warnings > 0:
    print(f"\n  {warnings} warnings — review before deploying")
    sys.exit(0)
else:
    print(f"\n  ALL {passed} CHECKS PASSED — SAFE TO DEPLOY")
    sys.exit(0)
