#!/usr/bin/env python3
"""
Reduce Kaggle notebook output from ~75,000 lines to ~500 lines.

Problem: 
  tqdm updates every single batch → 1,204 train + ~300 val = ~1,500 lines PER EPOCH
  28 epochs = ~42,000+ lines → browser crashes

Solution (from Kaggle community + tqdm docs):
  1. Use tqdm.auto (renders as widget in notebook, text in terminal)
  2. Add miniters=50 so the bar only updates every 50 batches  
  3. Add mininterval=5.0 so the bar only updates every 5 seconds
  
  Result: ~24 tqdm updates per epoch (1204/50) → 28 epochs = ~672 lines total
  Reduction: 75,000 → ~700 lines (99% reduction)
"""
import json, sys

sys.stdout.reconfigure(encoding='utf-8')
NB_PATH = r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\livertumor-model.ipynb'

nb = json.load(open(NB_PATH, encoding='utf-8'))
changes = 0

# ─────────────────────────────────────────────
# STEP 1: Fix the tqdm import (Cell 6)
# ─────────────────────────────────────────────
cell6_src = ''.join(nb['cells'][6]['source'])
old_import = "from tqdm import tqdm"
new_import = "from tqdm.auto import tqdm  # v7.2: auto selects notebook widget or text mode"

if old_import in cell6_src:
    cell6_src = cell6_src.replace(old_import, new_import)
    lines = cell6_src.split('\n')
    nb['cells'][6]['source'] = [l + '\n' if i < len(lines)-1 else l for i, l in enumerate(lines)]
    changes += 1
    print("✅ Fixed tqdm import: tqdm → tqdm.auto")

# ─────────────────────────────────────────────
# STEP 2: Add miniters + mininterval to training tqdm (Cell 12)
# ─────────────────────────────────────────────
cell12_src = ''.join(nb['cells'][12]['source'])

# Fix TRAIN tqdm
old_train_tqdm = 'tqdm(train_ld, desc=f"E{epoch:02d} Train", leave=False)'
new_train_tqdm = 'tqdm(train_ld, desc=f"E{epoch:02d} Train", leave=False, miniters=50, mininterval=10.0)'

if old_train_tqdm in cell12_src:
    cell12_src = cell12_src.replace(old_train_tqdm, new_train_tqdm)
    changes += 1
    print("✅ Fixed training tqdm: added miniters=50, mininterval=10.0")

# Fix VALIDATION tqdm
old_val_tqdm = 'tqdm(val_ld, desc=f"E{epoch:02d} Val", leave=False)'
new_val_tqdm = 'tqdm(val_ld, desc=f"E{epoch:02d} Val", leave=False, miniters=50, mininterval=10.0)'

if old_val_tqdm in cell12_src:
    cell12_src = cell12_src.replace(old_val_tqdm, new_val_tqdm)
    changes += 1
    print("✅ Fixed validation tqdm: added miniters=50, mininterval=10.0")

# Write back Cell 12
lines = cell12_src.split('\n')
nb['cells'][12]['source'] = [l + '\n' if i < len(lines)-1 else l for i, l in enumerate(lines)]

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# ─────────────────────────────────────────────
# ESTIMATE OUTPUT REDUCTION
# ─────────────────────────────────────────────
train_batches = 1204
val_batches = 302  # approx
epochs = 35  # expected

old_lines_per_epoch = (train_batches + val_batches) * 2  # each tqdm update = ~2 lines in text mode
new_lines_per_epoch = (train_batches // 50 + val_batches // 50) * 2 + 20  # ~20 for epoch summary + health

print(f"\n{'='*60}")
print(f"  OUTPUT REDUCTION ESTIMATE")
print(f"{'='*60}")
print(f"  Before: ~{old_lines_per_epoch} lines/epoch × {epochs} epochs = ~{old_lines_per_epoch * epochs:,} lines")
print(f"  After:  ~{new_lines_per_epoch} lines/epoch × {epochs} epochs = ~{new_lines_per_epoch * epochs:,} lines")
print(f"  Reduction: {(1 - new_lines_per_epoch / old_lines_per_epoch) * 100:.0f}%")
print(f"{'='*60}")
print(f"\n✅ APPLIED ({changes} changes)")
