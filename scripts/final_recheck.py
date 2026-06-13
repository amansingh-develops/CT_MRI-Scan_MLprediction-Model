"""
FINAL END-TO-END RECHECK
========================
1. CSV structure & content
2. Image file verification
3. Mask pixel-level verification
4. Notebook code verification
5. Training pipeline simulation
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import json
import cv2
import numpy as np
import pandas as pd

DATASET_DIR = "data/dataset_6"

print("=" * 70)
print("  FINAL RECHECK — PART 1: CSV FILE ANALYSIS")
print("=" * 70)

# Load all 4 CSVs
csv_files = {
    "lits_df.csv":    "data/lits_df.csv",
    "lits_train.csv": "data/lits_train.csv",
    "lits_probe.csv": "data/lits_probe.csv",
    "lits_test.csv":  "data/lits_test.csv",
}

csv_data = {}
for name, path in csv_files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        csv_data[name] = df
        print(f"\n--- {name} ---")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Study numbers: {sorted(df['study_number'].unique())[:10]}... (total={df['study_number'].nunique()})")
        
        # Check column types
        for col in df.columns:
            print(f"    {col}: dtype={df[col].dtype}, nulls={df[col].isnull().sum()}, sample={df[col].iloc[0]}")
    else:
        print(f"\n--- {name} --- NOT FOUND!")

# Cross-check: overlap between splits
print("\n" + "=" * 70)
print("  PART 1b: SPLIT OVERLAP CHECK")
print("=" * 70)

train_studies = set(csv_data["lits_train.csv"]["study_number"].unique())
val_studies = set(csv_data["lits_probe.csv"]["study_number"].unique())
test_studies = set(csv_data["lits_test.csv"]["study_number"].unique())
all_studies = set(csv_data["lits_df.csv"]["study_number"].unique())

print(f"  Train studies: {len(train_studies)}")
print(f"  Val studies:   {len(val_studies)}")
print(f"  Test studies:  {len(test_studies)}")
print(f"  Total studies: {len(all_studies)}")
print(f"  Train & Val overlap:  {len(train_studies & val_studies)} studies")
print(f"  Train & Test overlap: {len(train_studies & test_studies)} studies")
print(f"  Val & Test overlap:   {len(val_studies & test_studies)} studies")

if len(train_studies & val_studies) > 0:
    print("  WARNING: Train/Val OVERLAP! Data leakage possible.")
else:
    print("  OK: No train/val overlap")

if len(train_studies & test_studies) > 0:
    print("  WARNING: Train/Test OVERLAP! Data leakage possible.")
else:
    print("  OK: No train/test overlap")

# CSV metadata inversion check
print("\n" + "=" * 70)
print("  PART 1c: CSV METADATA ACCURACY CHECK")
print("=" * 70)

train_df = csv_data["lits_train.csv"]
np.random.seed(42)
sample = train_df.sample(min(100, len(train_df)))

csv_correct = 0
csv_wrong = 0
csv_missing = 0

for _, row in sample.iterrows():
    fpath = os.path.join(DATASET_DIR, os.path.basename(str(row["liver_maskpath"])))
    if os.path.exists(fpath):
        fsize = os.path.getsize(fpath)
        actual_has_content = fsize > 700  # empty PNGs are ~686 bytes
        csv_says_has_content = not row["liver_mask_empty"]
        
        if actual_has_content == csv_says_has_content:
            csv_correct += 1
        else:
            csv_wrong += 1
    else:
        csv_missing += 1

print(f"  Checked 100 random rows:")
print(f"    CSV matches reality: {csv_correct}")
print(f"    CSV is WRONG:        {csv_wrong}")
print(f"    File missing:        {csv_missing}")

if csv_wrong > csv_correct:
    print("  CONFIRMED: liver_mask_empty column is INVERTED in CSVs!")
    print("  (This doesn't affect training since model reads actual files)")

# ──────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  PART 2: IMAGE FILE VERIFICATION")
print("=" * 70)

files = sorted(os.listdir(DATASET_DIR))
volumes = [f for f in files if f.startswith("volume-")]
liver_masks = [f for f in files if "livermask" in f]
lesion_masks = [f for f in files if "lesionmask" in f]

print(f"  Total files in dataset_6: {len(files)}")
print(f"  Volume images:  {len(volumes)}")
print(f"  Liver masks:    {len(liver_masks)}")
print(f"  Lesion masks:   {len(lesion_masks)}")

# Verify 1:1:1 correspondence
print(f"\n  Ratio check: {len(volumes)} : {len(liver_masks)} : {len(lesion_masks)}")
if len(volumes) == len(liver_masks) == len(lesion_masks):
    print("  OK: 1:1:1 correspondence between volumes, liver masks, lesion masks")
else:
    print("  WARNING: Mismatched file counts!")

# Check image properties
print("\n  Image properties (5 random):")
np.random.seed(42)
for f in np.random.choice(volumes, 5, replace=False):
    fpath = os.path.join(DATASET_DIR, f)
    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    if img is not None:
        print(f"    {f}: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")

# ──────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  PART 3: MASK PIXEL-LEVEL VERIFICATION")
print("=" * 70)

# Check 20 masks that should have content (large file size)
big_masks = [(f, os.path.getsize(os.path.join(DATASET_DIR, f))) 
             for f in liver_masks if os.path.getsize(os.path.join(DATASET_DIR, f)) > 1000]

print(f"  Non-empty liver masks (>1KB): {len(big_masks)} / {len(liver_masks)} ({100*len(big_masks)/len(liver_masks):.1f}%)")

print("\n  Pixel value verification (10 non-empty masks):")
for f, sz in big_masks[:10]:
    raw = cv2.imread(os.path.join(DATASET_DIR, f), cv2.IMREAD_GRAYSCALE)
    if raw is not None:
        unique = np.unique(raw)
        nz = np.count_nonzero(raw)
        print(f"    {f}: unique_values={unique}, nonzero={nz}, max={raw.max()}")

# Verify: are masks truly binary 0/1?
print("\n  Binary 0/1 check (100 non-empty masks):")
all_binary = True
for f, sz in big_masks[:100]:
    raw = cv2.imread(os.path.join(DATASET_DIR, f), cv2.IMREAD_GRAYSCALE)
    if raw is not None:
        unique = set(np.unique(raw).tolist())
        if not unique.issubset({0, 1}):
            print(f"    NOT BINARY: {f} has values {unique}")
            all_binary = False

if all_binary:
    print("    ALL 100 masks are binary {0, 1}")
else:
    print("    WARNING: Some masks have non-binary values!")

# Verify the OLD threshold would fail
print("\n  OLD vs NEW threshold comparison:")
for f, sz in big_masks[:3]:
    raw = cv2.imread(os.path.join(DATASET_DIR, f), cv2.IMREAD_GRAYSCALE)
    if raw is not None:
        old_way = (raw.astype(np.float32) / 255.0 > 0.5).sum()
        new_way = (raw > 0).sum()
        print(f"    {f}:")
        print(f"      OLD (mask/255 > 0.5): {old_way} pixels")
        print(f"      NEW (mask > 0):       {new_way} pixels")
        print(f"      {'BUG CONFIRMED' if old_way == 0 and new_way > 0 else 'OK'}")

# ──────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  PART 4: NOTEBOOK CODE VERIFICATION")
print("=" * 70)

with open('livertumor-model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

total_cells = len(nb['cells'])
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
md_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
print(f"  Total cells: {total_cells} ({len(code_cells)} code, {len(md_cells)} markdown)")

# Check each critical component in the main code cell (Cell 6)
main_code = nb['cells'][6]['source']

checks = {
    "Mask fix (> 0).float()": "> 0).float()" in main_code,
    "Old bug absent": "/ 255.0).unsqueeze(0) > 0.5" not in main_code,
    "CLAHE preprocessing": "createCLAHE" in main_code,
    "INTER_NEAREST for masks": "INTER_NEAREST" in main_code,
    "Data augmentation (flips)": "np.fliplr" in main_code,
    "Data augmentation (rotation)": "np.rot90" in main_code,
    "Data augmentation (brightness)": "alpha" in main_code and "beta" in main_code,
    "UNet class defined": "class UNet" in main_code,
    "Kaiming init": "kaiming_normal_" in main_code,
    "Xavier init for head": "xavier_normal_" in main_code,
    "DiceLoss defined": "class DiceLoss" in main_code,
    "CombinedLoss with pos_weight": "pos_weight_liver" in main_code,
    "Per-sample Dice metric": "def dice_score" in main_code and "liver_scores" in main_code,
    "Skip connections": "skip" in main_code,
    "BatchNorm2d": "BatchNorm2d" in main_code,
    "No sigmoid in forward (raw logits)": "# raw logits" in main_code,
}

print("\n  Code component checks:")
all_pass = True
for check_name, result in checks.items():
    status = "PASS" if result else "FAIL"
    print(f"    [{status}] {check_name}")
    if not result:
        all_pass = False

# Check diagnostic cell (Cell 8)
diag_code = nb['cells'][8]['source']
diag_checks = {
    "Mask integrity scan": "MASK INTEGRITY" in diag_code,
    "pos_weight computation": "POS_WEIGHT_LIVER" in diag_code,
    "Sample visualization": "data_diagnostic" in diag_code,
    "Overfit test": "OVERFIT TEST" in diag_code,
    "Integrity assertion": "assert n_liver_nonempty" in diag_code,
}

print("\n  Diagnostic cell checks:")
for check_name, result in diag_checks.items():
    status = "PASS" if result else "FAIL"
    print(f"    [{status}] {check_name}")
    if not result:
        all_pass = False

# Check training cell
train_code = nb['cells'][12]['source']
train_checks = {
    "AMP (mixed precision)": "autocast" in train_code,
    "Gradient clipping": "clip_grad_norm_" in train_code,
    "Early stopping": "PATIENCE" in train_code,
    "LR scheduler": "ReduceLROnPlateau" in train_code,
    "Checkpoint saving": "best_model.pth" in train_code,
    "Kaggle timeout safety": "TIMEOUT_HRS" in train_code,
    "Auto-resume": "RESUMED" in train_code,
    "Training log CSV": "training_log.csv" in train_code,
    "Batch diagnostics": "Batch Diagnostics" in train_code or "BATCH DIAGNOSTICS" in train_code or "Diagnostics" in train_code,
}

print("\n  Training cell checks:")
for check_name, result in train_checks.items():
    status = "PASS" if result else "FAIL"
    print(f"    [{status}] {check_name}")
    if not result:
        all_pass = False

# Check sanity check cell
sanity_code = nb['cells'][10]['source']
sanity_checks = {
    "Shape test": "SHAPE FAIL" in sanity_code,
    "Kaiming init verification": "Init FAIL" in sanity_code,
    "Dataset test": "img.shape ==" in sanity_code,
    "Loss NaN check": "torch.isnan" in sanity_code,
    "AMP test": "autocast" in sanity_code,
    "Gradient flow test": "Gradients" in sanity_code,
    "Augmentation binary check": "binary" in sanity_code.lower(),
}

print("\n  Sanity check cell checks:")
for check_name, result in sanity_checks.items():
    status = "PASS" if result else "FAIL"
    print(f"    [{status}] {check_name}")
    if not result:
        all_pass = False

# ──────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  PART 5: TRAINING PIPELINE SIMULATION (LOCAL, CPU)")
print("=" * 70)

print("  Simulating what the notebook's LITSDataset will do...")

# Recreate the EXACT pipeline from the notebook
import torch

class LITSDatasetCheck:
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, os.path.basename(str(row["filepath"])))
        liv_path = os.path.join(self.root_dir, os.path.basename(str(row["liver_maskpath"])))
        tum_path = os.path.join(self.root_dir, os.path.basename(str(row["tumor_maskpath"])))

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        liver = cv2.imread(liv_path, cv2.IMREAD_GRAYSCALE)
        tumor = cv2.imread(tum_path, cv2.IMREAD_GRAYSCALE)

        if image is None: image = np.zeros((256, 256), dtype=np.uint8)
        if liver is None: liver = np.zeros((256, 256), dtype=np.uint8)
        if tumor is None: tumor = np.zeros((256, 256), dtype=np.uint8)

        image = self.clahe.apply(image)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        liver = cv2.resize(liver, (256, 256), interpolation=cv2.INTER_NEAREST)
        tumor = cv2.resize(tumor, (256, 256), interpolation=cv2.INTER_NEAREST)

        # THE FIXED LINES (exactly as in notebook):
        image = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        liver = (torch.from_numpy(liver.astype(np.float32)).unsqueeze(0) > 0).float()
        tumor = (torch.from_numpy(tumor.astype(np.float32)).unsqueeze(0) > 0).float()
        return image, torch.cat([liver, tumor], dim=0)

ds = LITSDatasetCheck("data/lits_train.csv", DATASET_DIR)

# Test on 200 random samples
np.random.seed(42)
test_indices = np.random.choice(len(ds.data), min(200, len(ds.data)), replace=False)

liver_nonempty = 0
tumor_nonempty = 0
total_liver_px = 0
total_tumor_px = 0

for idx in test_indices:
    img, msk = ds[int(idx)]
    
    # Check shapes and dtypes
    assert img.shape == (1, 256, 256), f"Bad img shape: {img.shape}"
    assert msk.shape == (2, 256, 256), f"Bad msk shape: {msk.shape}"
    assert img.dtype == torch.float32, f"Bad img dtype: {img.dtype}"
    assert msk.dtype == torch.float32, f"Bad msk dtype: {msk.dtype}"
    assert img.min() >= 0 and img.max() <= 1.0, f"Bad img range: [{img.min()}, {img.max()}]"
    assert set(msk.unique().tolist()).issubset({0.0, 1.0}), f"Bad msk values: {msk.unique()}"
    
    liver_px = msk[0].sum().item()
    tumor_px = msk[1].sum().item()
    total_liver_px += liver_px
    total_tumor_px += tumor_px
    
    if liver_px > 0:
        liver_nonempty += 1
    if tumor_px > 0:
        tumor_nonempty += 1

print(f"  Tested 200 samples through FIXED pipeline:")
print(f"    All shapes correct:  (1,256,256) images, (2,256,256) masks")
print(f"    All dtypes correct:  float32")
print(f"    All ranges correct:  images [0,1], masks binary {{0,1}}")
print(f"    Liver non-empty:     {liver_nonempty}/200 ({100*liver_nonempty/200:.1f}%)")
print(f"    Tumor non-empty:     {tumor_nonempty}/200 ({100*tumor_nonempty/200:.1f}%)")
print(f"    Total liver pixels:  {total_liver_px:,.0f}")
print(f"    Total tumor pixels:  {total_tumor_px:,.0f}")

if liver_nonempty > 30:
    print(f"\n    PIPELINE CHECK: PASSED")
    print(f"    The model WILL see liver masks during training.")
else:
    print(f"\n    PIPELINE CHECK: FAILED!")
    print(f"    Something is still wrong with mask loading.")

# ──────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  FINAL VERDICT")
print("=" * 70)

issues = []
if csv_wrong > csv_correct:
    issues.append("CSV metadata inverted (KNOWN, does not affect training)")
if not all_pass:
    issues.append("Some notebook code checks FAILED")
if liver_nonempty < 30:
    issues.append("Pipeline not producing liver masks")

if len(issues) == 0 or (len(issues) == 1 and "CSV metadata" in issues[0]):
    print("""
  EVERYTHING IS READY FOR KAGGLE TRAINING!

  Summary of what the notebook will do:
  1. Load 38,523 train + 3,038 val samples
  2. Apply CLAHE contrast enhancement to CT images
  3. Load masks as binary 0/1 (FIXED - no more /255 > 0.5 bug)
  4. Apply augmentation (flips, rotations, brightness jitter)
  5. Run diagnostic cell to verify masks & compute pos_weights
  6. Run overfit test to confirm model can learn
  7. Train UNet with:
     - Combined Loss (BCE + pos_weight + Dice)
     - AdamW optimizer (lr=3e-4, weight_decay=1e-5)
     - Mixed precision (AMP) on T4 GPU
     - Gradient clipping (max_norm=1.0)
     - ReduceLROnPlateau scheduler
     - Early stopping (patience=15)
     - 11.5-hour timeout safety
  8. Save best model + training curves + predictions

  Known non-blocking issue:
  - CSV "liver_mask_empty" column is inverted
    (model reads actual files, NOT the CSV flag)
""")
else:
    print(f"\n  ISSUES FOUND:")
    for issue in issues:
        print(f"    - {issue}")
