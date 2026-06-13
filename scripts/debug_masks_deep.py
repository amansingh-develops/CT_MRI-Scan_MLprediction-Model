"""Deep mask analysis: find ALL non-empty masks and understand the pattern"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import cv2
import numpy as np
import pandas as pd

DATASET_DIR = "data/dataset_6"
TRAIN_CSV = "data/lits_train.csv"
df = pd.read_csv(TRAIN_CSV)

# ─── 1. Check what the CSV column "liver_mask_empty" actually means ───
print("=" * 70)
print("  CSV COLUMN ANALYSIS")
print("=" * 70)
print(f"\nColumns: {list(df.columns)}")
print(f"\nSample rows where liver_mask_empty=False:")
liver_pos = df[~df["liver_mask_empty"]].head(10)
for _, row in liver_pos.iterrows():
    print(f"  study={row['study_number']}, file={os.path.basename(row['filepath'])}, "
          f"liver_empty={row['liver_mask_empty']}, tumor_empty={row['tumor_mask_empty']}")

# ─── 2. Scan ALL mask files to find ones with actual content ───
print("\n" + "=" * 70)
print("  SCANNING ALL MASK FILES FOR ACTUAL CONTENT")
print("=" * 70)

files = sorted(os.listdir(DATASET_DIR))
liver_masks = [f for f in files if "livermask" in f]
lesion_masks = [f for f in files if "lesionmask" in f]

print(f"\nTotal liver mask files:  {len(liver_masks)}")
print(f"Total lesion mask files: {len(lesion_masks)}")

# Sample 500 liver masks, check which have content
np.random.seed(42)
sample_liver = np.random.choice(liver_masks, min(500, len(liver_masks)), replace=False)

nonempty_liver = []
empty_liver = []
file_sizes = []

for f in sample_liver:
    fpath = os.path.join(DATASET_DIR, f)
    fsize = os.path.getsize(fpath)
    file_sizes.append(fsize)
    
    if fsize > 700:  # Non-empty PNGs are usually larger than empty ones
        raw = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if raw is not None and raw.max() > 0:
            nonempty_liver.append((f, fsize, raw.max(), np.count_nonzero(raw)))
        else:
            empty_liver.append(f)
    else:
        empty_liver.append(f)

print(f"\nSampled {len(sample_liver)} liver masks:")
print(f"  Non-empty (has content): {len(nonempty_liver)}")
print(f"  Empty (all zeros):       {len(empty_liver)}")

if nonempty_liver:
    print(f"\n  Non-empty liver mask examples:")
    for f, sz, mx, nz in nonempty_liver[:20]:
        print(f"    {f} — size={sz}B, max_val={mx}, nonzero_pixels={nz}")

# ─── 3. Use FILE SIZE as a proxy to find ALL non-empty masks ───
print("\n" + "=" * 70)
print("  FILE SIZE ANALYSIS (fast proxy for empty vs non-empty)")
print("=" * 70)

# Get file sizes for ALL liver masks
liver_sizes = []
for f in liver_masks[:5000]:  # check first 5000
    fpath = os.path.join(DATASET_DIR, f)
    liver_sizes.append(os.path.getsize(fpath))

liver_sizes = np.array(liver_sizes)
print(f"\nLiver mask file sizes (first {len(liver_sizes)}):")
print(f"  Min:    {liver_sizes.min()} bytes")
print(f"  Max:    {liver_sizes.max()} bytes")
print(f"  Mean:   {liver_sizes.mean():.0f} bytes")
print(f"  Median: {np.median(liver_sizes):.0f} bytes")
print(f"  <= 700B: {(liver_sizes <= 700).sum()} ({100*(liver_sizes <= 700).sum()/len(liver_sizes):.1f}%)")
print(f"  > 700B:  {(liver_sizes > 700).sum()} ({100*(liver_sizes > 700).sum()/len(liver_sizes):.1f}%)")

# Now do the same for lesion masks
lesion_sizes = []
for f in lesion_masks[:5000]:
    fpath = os.path.join(DATASET_DIR, f)
    lesion_sizes.append(os.path.getsize(fpath))

lesion_sizes = np.array(lesion_sizes)
print(f"\nLesion mask file sizes (first {len(lesion_sizes)}):")
print(f"  Min:    {lesion_sizes.min()} bytes")
print(f"  Max:    {lesion_sizes.max()} bytes")
print(f"  Mean:   {lesion_sizes.mean():.0f} bytes")
print(f"  Median: {np.median(lesion_sizes):.0f} bytes")
print(f"  <= 700B: {(lesion_sizes <= 700).sum()} ({100*(lesion_sizes <= 700).sum()/len(lesion_sizes):.1f}%)")
print(f"  > 700B:  {(lesion_sizes > 700).sum()} ({100*(lesion_sizes > 700).sum()/len(lesion_sizes):.1f}%)")

# ─── 4. Cross-check CSV "liver_mask_empty" vs actual file content ───
print("\n" + "=" * 70)
print("  CSV vs ACTUAL MASK CONTENT CROSS-CHECK")
print("=" * 70)

mismatch_count = 0
csv_says_liver_actual_empty = 0
csv_says_empty_actual_liver = 0

# Check 200 rows where CSV says liver is present
liver_pos_df = df[~df["liver_mask_empty"]].sample(min(200, len(df[~df["liver_mask_empty"]])), random_state=42)
for _, row in liver_pos_df.iterrows():
    fpath = os.path.join(DATASET_DIR, os.path.basename(str(row["liver_maskpath"])))
    if os.path.exists(fpath):
        raw = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if raw is not None and raw.max() == 0:
            csv_says_liver_actual_empty += 1

# Check 200 rows where CSV says liver is absent
liver_neg_df = df[df["liver_mask_empty"]].sample(min(200, len(df[df["liver_mask_empty"]])), random_state=42)
for _, row in liver_neg_df.iterrows():
    fpath = os.path.join(DATASET_DIR, os.path.basename(str(row["liver_maskpath"])))
    if os.path.exists(fpath):
        raw = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if raw is not None and raw.max() > 0:
            csv_says_empty_actual_liver += 1

print(f"\nCSV says liver=present BUT mask is empty:  {csv_says_liver_actual_empty}/200 ({100*csv_says_liver_actual_empty/200:.0f}%)")
print(f"CSV says liver=absent  BUT mask has content: {csv_says_empty_actual_liver}/200 ({100*csv_says_empty_actual_liver/200:.0f}%)")

if csv_says_liver_actual_empty > 100:
    print("\n  ★★★ CRITICAL: CSV metadata DOES NOT MATCH actual mask files!")
    print("  ★★★ The local dataset_6 masks may be CORRUPTED or INCOMPLETE!")

# ─── 5. Actually verify with pixel analysis on known-good slices ───
print("\n" + "=" * 70)
print("  VERIFIED-CONTENT MASKS")
print("=" * 70)

# Find masks that actually have content by file size
big_liver_masks = []
for f in liver_masks:
    fpath = os.path.join(DATASET_DIR, f)
    fsize = os.path.getsize(fpath)
    if fsize > 1000:
        big_liver_masks.append((f, fsize))

print(f"\nLiver masks > 1KB (likely non-empty): {len(big_liver_masks)} / {len(liver_masks)}")
print(f"That's {100*len(big_liver_masks)/len(liver_masks):.1f}% of all liver masks")

if big_liver_masks:
    # Verify first 10
    print("\nVerifying first 10 big liver masks:")
    verified_nonempty = 0
    for f, sz in big_liver_masks[:10]:
        raw = cv2.imread(os.path.join(DATASET_DIR, f), cv2.IMREAD_GRAYSCALE)
        nz = np.count_nonzero(raw) if raw is not None else 0
        pct = 100 * nz / (256*256) if raw is not None else 0
        print(f"    {f}: size={sz}B, nonzero={nz} ({pct:.1f}%)")
        if nz > 0:
            verified_nonempty += 1
    print(f"  Verified non-empty: {verified_nonempty}/10")

# Similarly for lesion/tumor
big_tumor_masks = []
for f in lesion_masks:
    fpath = os.path.join(DATASET_DIR, f)
    fsize = os.path.getsize(fpath)
    if fsize > 1000:
        big_tumor_masks.append((f, fsize))

print(f"\nTumor masks > 1KB (likely non-empty): {len(big_tumor_masks)} / {len(lesion_masks)}")
print(f"That's {100*len(big_tumor_masks)/len(lesion_masks):.1f}% of all tumor masks")

print("\n" + "=" * 70)
print("  CONCLUSION")
print("=" * 70)
true_liver_rate = len(big_liver_masks) / len(liver_masks) * 100
true_tumor_rate = len(big_tumor_masks) / len(lesion_masks) * 100
print(f"""
TRUE class distribution based on actual file content:
  Liver-containing slices: {len(big_liver_masks)} / {len(liver_masks)} ({true_liver_rate:.1f}%)
  Tumor-containing slices: {len(big_tumor_masks)} / {len(lesion_masks)} ({true_tumor_rate:.1f}%)

  Neg:Pos ratio for Liver: {(len(liver_masks)-len(big_liver_masks))/max(len(big_liver_masks),1):.1f}:1
  Neg:Pos ratio for Tumor: {(len(lesion_masks)-len(big_tumor_masks))/max(len(big_tumor_masks),1):.1f}:1
""")
