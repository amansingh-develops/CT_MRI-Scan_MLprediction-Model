"""Debug: investigate why ALL masks are reading as zeros"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import cv2
import numpy as np
import pandas as pd

DATASET_DIR = "data/dataset_6"
TRAIN_CSV = "data/lits_train.csv"
df = pd.read_csv(TRAIN_CSV)

# Pick a row CSV says has liver AND tumor
liver_rows = df[~df["liver_mask_empty"] & ~df["tumor_mask_empty"]]
print(f"CSV says {len(liver_rows)} rows have BOTH liver and tumor")

# Check first 5 such rows
for i in range(min(5, len(liver_rows))):
    row = liver_rows.iloc[i]
    
    # What paths does CSV give us?
    csv_liver_path = str(row["liver_maskpath"])
    csv_tumor_path = str(row["tumor_maskpath"])
    csv_img_path = str(row["filepath"])
    
    # What we're loading
    local_liver = os.path.join(DATASET_DIR, os.path.basename(csv_liver_path))
    local_tumor = os.path.join(DATASET_DIR, os.path.basename(csv_tumor_path))
    local_img = os.path.join(DATASET_DIR, os.path.basename(csv_img_path))
    
    print(f"\n--- Row {i} (study={row['study_number']}) ---")
    print(f"  CSV liver path: {csv_liver_path}")
    print(f"  Local liver:    {local_liver}")
    print(f"  Exists:         {os.path.exists(local_liver)}")
    
    if os.path.exists(local_liver):
        raw = cv2.imread(local_liver, cv2.IMREAD_UNCHANGED)
        gray = cv2.imread(local_liver, cv2.IMREAD_GRAYSCALE)
        print(f"  Raw shape:  {raw.shape if raw is not None else 'NONE'}")
        print(f"  Raw dtype:  {raw.dtype if raw is not None else 'NONE'}")
        if raw is not None:
            print(f"  Raw min/max: {raw.min()}/{raw.max()}")
            print(f"  Raw unique:  {np.unique(raw)[:10]}")
            print(f"  Raw sum:     {raw.sum()}")
        if gray is not None:
            print(f"  Gray min/max: {gray.min()}/{gray.max()}")
            print(f"  Gray sum:     {gray.sum()}")
    
    if os.path.exists(local_tumor):
        raw_t = cv2.imread(local_tumor, cv2.IMREAD_UNCHANGED)
        if raw_t is not None:
            print(f"  Tumor raw shape: {raw_t.shape}")
            print(f"  Tumor min/max: {raw_t.min()}/{raw_t.max()}")
            print(f"  Tumor unique:  {np.unique(raw_t)[:10]}")

# Also check: what are the actual mask filenames?
print("\n\n--- Checking mask filename patterns ---")
files = os.listdir(DATASET_DIR)
liver_masks = sorted([f for f in files if "livermask" in f])[:5]
lesion_masks = sorted([f for f in files if "lesionmask" in f])[:5]
print(f"  Sample liver mask files:  {liver_masks}")
print(f"  Sample lesion mask files: {lesion_masks}")

# Check what's actually IN a liver mask file
if liver_masks:
    test_path = os.path.join(DATASET_DIR, liver_masks[0])
    raw = cv2.imread(test_path, cv2.IMREAD_UNCHANGED)
    print(f"\n  First liver mask: {liver_masks[0]}")
    print(f"    Shape: {raw.shape if raw is not None else 'NONE'}")
    print(f"    Dtype: {raw.dtype if raw is not None else 'NONE'}")
    if raw is not None:
        print(f"    Min/Max: {raw.min()}/{raw.max()}")
        print(f"    Unique values: {np.unique(raw)}")
        print(f"    Non-zero count: {np.count_nonzero(raw)}")
        print(f"    Size: {os.path.getsize(test_path)} bytes")

# Check a KNOWN good mask (volume-0_0 which CSV says has liver)
print("\n\n--- Checking specific volume-0 masks ---")
for suffix in ["_0", "_1", "_2", "_50", "_100", "_200"]:
    fname = f"segmentation-0_livermask{suffix}.png"
    fpath = os.path.join(DATASET_DIR, fname)
    if os.path.exists(fpath):
        raw = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        nonzero = np.count_nonzero(raw) if raw is not None else -1
        fsize = os.path.getsize(fpath)
        print(f"  {fname}: shape={raw.shape if raw is not None else 'N/A'}, nonzero={nonzero}, size={fsize}B")
    else:
        print(f"  {fname}: NOT FOUND")

# Check if volume files have a different naming pattern
print("\n\n--- Looking for segmentation files for volume-2 ---")
vol2_files = sorted([f for f in files if f.startswith("segmentation-2")])[:10]
print(f"  First 10: {vol2_files}")
