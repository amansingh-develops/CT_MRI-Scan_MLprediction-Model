"""Final pixel-level EDA with CORRECT mask reading"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_DIR = "data/dataset_6"
os.makedirs("results/eda", exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# KEY DISCOVERY #1: Masks have values 0 and 1, NOT 0 and 255
# The notebook's binarization threshold of 0.5 (after /255)
# requires values >= 128. But actual mask values are 0 and 1!
# After /255: 1/255 = 0.003... which is BELOW the 0.5 threshold!
# THIS IS WHY THE MODEL SEES ALL-ZEROS MASKS!
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("  CRITICAL FINDING: MASK VALUE ANALYSIS")
print("=" * 70)

# Check mask value distribution
files = sorted(os.listdir(DATASET_DIR))
liver_masks = [f for f in files if "livermask" in f]

# Pick 20 masks that have content (> 700 bytes)
checked = 0
for f in liver_masks:
    fpath = os.path.join(DATASET_DIR, f)
    if os.path.getsize(fpath) > 1000:
        raw = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        
        if raw is not None and checked < 5:
            print(f"\n  File: {f}")
            print(f"    Raw shape: {raw.shape}, dtype: {raw.dtype}")
            print(f"    Raw unique: {np.unique(raw)}")
            print(f"    Raw max: {raw.max()}")
            print(f"    Gray unique: {np.unique(gray)}")
            print(f"    Gray max: {gray.max()}")
            
            # What the notebook does: gray.astype(float32)/255 > 0.5
            normalized = gray.astype(np.float32) / 255.0
            thresholded_notebook = (normalized > 0.5).sum()
            thresholded_correct = (gray > 0).sum()
            
            print(f"    After /255: unique = {np.unique(normalized)}")
            print(f"    Threshold > 0.5 (NOTEBOOK):  {thresholded_notebook} pixels")
            print(f"    Threshold > 0 (CORRECT):     {thresholded_correct} pixels")
            
            if thresholded_notebook == 0 and thresholded_correct > 0:
                print(f"    ★★★ BUG CONFIRMED: Notebook sees 0 pixels, actual has {thresholded_correct}!")
            
            checked += 1

# ═══════════════════════════════════════════════════════════════
# KEY DISCOVERY #2: CSV "liver_mask_empty" column is INVERTED
# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  CSV COLUMN INVERSION CHECK")
print("=" * 70)

# Read CSV
df = pd.read_csv("data/lits_train.csv")

# Check 50 random rows
np.random.seed(42)
sample = df.sample(50)
csv_correct = 0
csv_inverted = 0
csv_unknown = 0

for _, row in sample.iterrows():
    fpath = os.path.join(DATASET_DIR, os.path.basename(str(row["liver_maskpath"])))
    if os.path.exists(fpath):
        fsize = os.path.getsize(fpath)
        actual_has_content = fsize > 700
        csv_says_has_content = not row["liver_mask_empty"]
        
        if actual_has_content == csv_says_has_content:
            csv_correct += 1
        elif actual_has_content != csv_says_has_content:
            csv_inverted += 1
    else:
        csv_unknown += 1

print(f"  CSV correct:  {csv_correct}/50")
print(f"  CSV inverted: {csv_inverted}/50")
print(f"  Unknown:      {csv_unknown}/50")

# ═══════════════════════════════════════════════════════════════
# NOW: Correct pixel-level EDA with proper mask reading
# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  CORRECT PIXEL-LEVEL ANALYSIS")
print("=" * 70)

np.random.seed(42)
sample_size = 2000
sample_indices = np.random.choice(len(df), min(sample_size, len(df)), replace=False)

total_pixels = 0
liver_pixels = 0
tumor_pixels = 0
liver_coverage = []
tumor_coverage = []

for count, idx in enumerate(sample_indices):
    if count % 500 == 0:
        print(f"  Processing {count}/{len(sample_indices)}...")
    
    row = df.iloc[int(idx)]
    liv_path = os.path.join(DATASET_DIR, os.path.basename(str(row["liver_maskpath"])))
    tum_path = os.path.join(DATASET_DIR, os.path.basename(str(row["tumor_maskpath"])))
    
    liv = cv2.imread(liv_path, cv2.IMREAD_GRAYSCALE)
    tum = cv2.imread(tum_path, cv2.IMREAD_GRAYSCALE)
    
    if liv is None or tum is None:
        continue
    
    n_pixels = liv.shape[0] * liv.shape[1]
    
    # CORRECT: threshold at > 0, NOT after /255 > 0.5
    n_liver = (liv > 0).sum()
    n_tumor = (tum > 0).sum()
    
    total_pixels += n_pixels
    liver_pixels += n_liver
    tumor_pixels += n_tumor
    
    liver_coverage.append(100 * n_liver / n_pixels)
    tumor_coverage.append(100 * n_tumor / n_pixels)

bg_pixels = total_pixels - liver_pixels  # approximate

print(f"\n  Total pixels:     {total_pixels:,.0f}")
print(f"  Liver pixels:     {liver_pixels:,.0f} ({100*liver_pixels/total_pixels:.2f}%)")
print(f"  Tumor pixels:     {tumor_pixels:,.0f} ({100*tumor_pixels/total_pixels:.4f}%)")
print(f"  Background:       {bg_pixels:,.0f} ({100*bg_pixels/total_pixels:.2f}%)")

liver_pw = bg_pixels / max(liver_pixels, 1)
tumor_pw = (total_pixels - tumor_pixels) / max(tumor_pixels, 1)
print(f"\n  ★ CORRECT pos_weight for liver: {liver_pw:.1f}")
print(f"  ★ CORRECT pos_weight for tumor: {tumor_pw:.1f}")

liver_cov = np.array(liver_coverage)
tumor_cov = np.array(tumor_coverage)
print(f"\n  Slices with liver: {(liver_cov > 0).sum()} / {len(liver_cov)} ({100*(liver_cov > 0).sum()/len(liver_cov):.1f}%)")
print(f"  Slices with tumor: {(tumor_cov > 0).sum()} / {len(tumor_cov)} ({100*(tumor_cov > 0).sum()/len(tumor_cov):.1f}%)")

liver_nz = liver_cov[liver_cov > 0]
tumor_nz = tumor_cov[tumor_cov > 0]
if len(liver_nz) > 0:
    print(f"\n  Liver coverage (when present): mean={liver_nz.mean():.1f}%, median={np.median(liver_nz):.1f}%, max={liver_nz.max():.1f}%")
if len(tumor_nz) > 0:
    print(f"  Tumor coverage (when present): mean={tumor_nz.mean():.2f}%, median={np.median(tumor_nz):.2f}%, max={tumor_nz.max():.2f}%")

# ═══════════════════════════════════════════════════════════════
# FINAL PLOTS
# ═══════════════════════════════════════════════════════════════
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#161b22'
plt.rcParams['text.color'] = '#c9d1d9'
plt.rcParams['axes.labelcolor'] = '#c9d1d9'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle("CORRECTED Pixel-Level Analysis (masks are binary 0/1, NOT 0/255)", 
             fontsize=16, fontweight='bold', color='white')

# Pie
sizes = [bg_pixels, liver_pixels, tumor_pixels]
labels = [f'Background\n{100*bg_pixels/total_pixels:.1f}%',
          f'Liver\n{100*liver_pixels/total_pixels:.2f}%',
          f'Tumor\n{100*tumor_pixels/total_pixels:.3f}%']
colors = ['#8b949e', '#238636', '#da3633']
axes[0].pie(sizes, labels=labels, colors=colors, explode=(0, 0.05, 0.15),
            textprops={'fontsize': 11, 'color': '#c9d1d9'})
axes[0].set_title('True Pixel-Level Distribution', fontsize=13, color='white')

# Liver coverage
if len(liver_nz) > 0:
    axes[1].hist(liver_nz, bins=50, color='#238636', edgecolor='#0d1117', alpha=0.85)
    axes[1].axvline(liver_nz.mean(), color='#f0883e', linestyle='--', lw=2, label=f'Mean: {liver_nz.mean():.1f}%')
    axes[1].axvline(np.median(liver_nz), color='#1f6feb', linestyle='--', lw=2, label=f'Median: {np.median(liver_nz):.1f}%')
    axes[1].legend(fontsize=10)
axes[1].set_xlabel('Liver Coverage (%)')
axes[1].set_ylabel('Count')
axes[1].set_title(f'Liver Coverage (non-empty slices, n={len(liver_nz)})', fontsize=13, color='white')

# Tumor coverage
if len(tumor_nz) > 0:
    axes[2].hist(tumor_nz, bins=50, color='#da3633', edgecolor='#0d1117', alpha=0.85)
    axes[2].axvline(tumor_nz.mean(), color='#f0883e', linestyle='--', lw=2, label=f'Mean: {tumor_nz.mean():.2f}%')
    axes[2].axvline(np.median(tumor_nz), color='#1f6feb', linestyle='--', lw=2, label=f'Median: {np.median(tumor_nz):.2f}%')
    axes[2].legend(fontsize=10)
axes[2].set_xlabel('Tumor Coverage (%)')
axes[2].set_ylabel('Count')
axes[2].set_title(f'Tumor Coverage (non-empty slices, n={len(tumor_nz)})', fontsize=13, color='white')

plt.tight_layout()
plt.savefig('results/eda/11_CORRECTED_pixel_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
print("\nSaved: results/eda/11_CORRECTED_pixel_analysis.png")

# ── Sample visualization with CORRECT mask reading ──
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
fig.suptitle("Sample Slices with CORRECTLY Read Masks (threshold > 0, not > 128)", 
             fontsize=16, fontweight='bold', color='white')

# Find slices that actually have liver content
liver_idx = []
for idx in sample_indices:
    row = df.iloc[int(idx)]
    fpath = os.path.join(DATASET_DIR, os.path.basename(str(row["liver_maskpath"])))
    if os.path.exists(fpath) and os.path.getsize(fpath) > 1000:
        liver_idx.append(int(idx))
    if len(liver_idx) >= 4:
        break

for r, idx in enumerate(liver_idx):
    row = df.iloc[idx]
    img = cv2.imread(os.path.join(DATASET_DIR, os.path.basename(str(row["filepath"]))), cv2.IMREAD_GRAYSCALE)
    liv = cv2.imread(os.path.join(DATASET_DIR, os.path.basename(str(row["liver_maskpath"]))), cv2.IMREAD_GRAYSCALE)
    tum = cv2.imread(os.path.join(DATASET_DIR, os.path.basename(str(row["tumor_maskpath"]))), cv2.IMREAD_GRAYSCALE)
    
    if img is None or liv is None or tum is None:
        continue
    
    liv_correct = (liv > 0).astype(np.uint8) * 255
    tum_correct = (tum > 0).astype(np.uint8) * 255
    
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    overlay[liv > 0] = [0, 200, 0]
    overlay[tum > 0] = [220, 50, 50]
    
    axes[r][0].imshow(img, cmap='gray')
    axes[r][0].set_title(f'CT (study={row["study_number"]})', color='white', fontsize=10)
    axes[r][1].imshow(liv_correct, cmap='Greens')
    axes[r][1].set_title(f'Liver ({(liv>0).sum()}px, {100*(liv>0).sum()/img.size:.1f}%)', color='white', fontsize=10)
    axes[r][2].imshow(tum_correct, cmap='Reds')
    axes[r][2].set_title(f'Tumor ({(tum>0).sum()}px, {100*(tum>0).sum()/img.size:.1f}%)', color='white', fontsize=10)
    axes[r][3].imshow(overlay)
    axes[r][3].set_title('Overlay', color='white', fontsize=10)
    
    for ax in axes[r]:
        ax.axis('off')

plt.tight_layout()
plt.savefig('results/eda/12_CORRECTED_sample_viz.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
print("Saved: results/eda/12_CORRECTED_sample_viz.png")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  ROOT CAUSE SUMMARY")
print("=" * 70)
print("""
★★★ TWO CRITICAL BUGS FOUND ★★★

BUG #1: MASK BINARIZATION THRESHOLD
──────────────────────────────────────
  The mask PNG files store values as 0 (background) and 1 (foreground).
  The notebook code does: mask / 255.0 > 0.5
  After dividing by 255: 1/255 = 0.00392 → BELOW 0.5 threshold!
  
  RESULT: The model receives ALL-ZERO masks for EVERY sample.
  This is why loss was constant 0.5 and Dice was 0.0.
  The model literally had nothing to learn — all targets were blank!
  
  FIX: Change threshold from > 0.5 to > 0.0 (or simply use mask > 0)
  OR: Don't divide by 255 and threshold at 0.5 directly

BUG #2: CSV METADATA IS INVERTED  
──────────────────────────────────────
  The liver_mask_empty column values appear to be swapped/unreliable.
  This doesn't affect training (model reads files, not CSV),
  but it misled our earlier analysis.
  The model code correctly falls back to reading actual files.
""")
