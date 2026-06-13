"""
Pixel-Level EDA — Analyze actual image and mask pixel distributions
=====================================================================
This is the critical EDA that tells us about PIXEL-level class imbalance
(which is what the model actually sees during training).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

os.makedirs("results/eda", exist_ok=True)

DATA_DIR = "data"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_6")
TRAIN_CSV = os.path.join(DATA_DIR, "lits_train.csv")

train_df = pd.read_csv(TRAIN_CSV)
print(f"Train CSV: {len(train_df)} samples")

# ═══════════════════════════════════════════════════════════════
# 1. PIXEL-LEVEL CLASS IMBALANCE ANALYSIS
#    Sample N images and compute actual pixel counts
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PIXEL-LEVEL ANALYSIS (sampling 2000 images)")
print("=" * 70)

N_SAMPLE = 2000
np.random.seed(42)
sample_indices = np.random.choice(len(train_df), min(N_SAMPLE, len(train_df)), replace=False)

total_pixels = 0
liver_pixels = 0
tumor_pixels = 0
background_pixels = 0

img_means = []
img_stds = []
img_mins = []
img_maxs = []

liver_coverage_per_slice = []  # % of pixels that are liver per slice
tumor_coverage_per_slice = []  # % of pixels that are tumor per slice

liver_nonempty_pixel_counts = []  # pixel count for liver-positive slices
tumor_nonempty_pixel_counts = []  # pixel count for tumor-positive slices

errors = 0

for count, idx in enumerate(sample_indices):
    if count % 500 == 0:
        print(f"  Processing {count}/{len(sample_indices)}...")
    
    row = train_df.iloc[int(idx)]
    img_path = os.path.join(DATASET_DIR, os.path.basename(str(row["filepath"])))
    liv_path = os.path.join(DATASET_DIR, os.path.basename(str(row["liver_maskpath"])))
    tum_path = os.path.join(DATASET_DIR, os.path.basename(str(row["tumor_maskpath"])))
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    liv = cv2.imread(liv_path, cv2.IMREAD_GRAYSCALE)
    tum = cv2.imread(tum_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or liv is None or tum is None:
        errors += 1
        continue
    
    # Image statistics
    img_float = img.astype(np.float32) / 255.0
    img_means.append(img_float.mean())
    img_stds.append(img_float.std())
    img_mins.append(img_float.min())
    img_maxs.append(img_float.max())
    
    # Mask binarization (same as in LITSDataset)
    liv_binary = (liv.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    tum_binary = (tum.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    
    n_pixels = img.shape[0] * img.shape[1]
    n_liver = liv_binary.sum()
    n_tumor = tum_binary.sum()
    n_bg = n_pixels - max(n_liver, n_tumor)  # background = not liver and not tumor
    
    total_pixels += n_pixels
    liver_pixels += n_liver
    tumor_pixels += n_tumor
    background_pixels += n_bg
    
    liver_coverage_per_slice.append(n_liver / n_pixels * 100)
    tumor_coverage_per_slice.append(n_tumor / n_pixels * 100)
    
    if n_liver > 0:
        liver_nonempty_pixel_counts.append(n_liver)
    if n_tumor > 0:
        tumor_nonempty_pixel_counts.append(n_tumor)

print(f"\n  Processed: {len(sample_indices) - errors} images, {errors} errors")

# ─── PIXEL-LEVEL STATISTICS ───
print("\n" + "-" * 50)
print("PIXEL-LEVEL CLASS BALANCE:")
print("-" * 50)
print(f"  Total pixels analyzed: {total_pixels:,.0f}")
print(f"  Background pixels:     {background_pixels:,.0f} ({100*background_pixels/total_pixels:.2f}%)")
print(f"  Liver pixels:          {liver_pixels:,.0f} ({100*liver_pixels/total_pixels:.2f}%)")
print(f"  Tumor pixels:          {tumor_pixels:,.0f} ({100*tumor_pixels/total_pixels:.2f}%)")

liver_pw = (total_pixels - liver_pixels) / max(liver_pixels, 1)
tumor_pw = (total_pixels - tumor_pixels) / max(tumor_pixels, 1)
print(f"\n  ★ Pixel-level pos_weight for liver: {liver_pw:.1f}")
print(f"  ★ Pixel-level pos_weight for tumor: {tumor_pw:.1f}")
print(f"  (These are the ACTUAL weights needed for BCE pos_weight)")

# ─── IMAGE STATISTICS ───
print("\n" + "-" * 50)
print("IMAGE PIXEL VALUE STATISTICS:")
print("-" * 50)
print(f"  Mean of means:  {np.mean(img_means):.4f}")
print(f"  Std of means:   {np.std(img_means):.4f}")
print(f"  Mean of stds:   {np.mean(img_stds):.4f}")
print(f"  Min of mins:    {np.min(img_mins):.4f}")
print(f"  Max of maxs:    {np.max(img_maxs):.4f}")
print(f"  Mean of mins:   {np.mean(img_mins):.4f}")
print(f"  Mean of maxs:   {np.mean(img_maxs):.4f}")

# ─── LIVER COVERAGE DISTRIBUTION ───
print("\n" + "-" * 50)
print("LIVER COVERAGE PER SLICE:")
print("-" * 50)
liver_arr = np.array(liver_coverage_per_slice)
print(f"  Slices with zero liver: {(liver_arr == 0).sum()} / {len(liver_arr)} ({100*(liver_arr == 0).sum()/len(liver_arr):.1f}%)")
print(f"  When liver IS present:")
liver_nonzero = liver_arr[liver_arr > 0]
if len(liver_nonzero) > 0:
    print(f"    Min coverage:    {liver_nonzero.min():.2f}%")
    print(f"    Max coverage:    {liver_nonzero.max():.2f}%")
    print(f"    Mean coverage:   {liver_nonzero.mean():.2f}%")
    print(f"    Median coverage: {np.median(liver_nonzero):.2f}%")

# ─── TUMOR COVERAGE DISTRIBUTION ───
print("\n" + "-" * 50)
print("TUMOR COVERAGE PER SLICE:")
print("-" * 50)
tumor_arr = np.array(tumor_coverage_per_slice)
print(f"  Slices with zero tumor: {(tumor_arr == 0).sum()} / {len(tumor_arr)} ({100*(tumor_arr == 0).sum()/len(tumor_arr):.1f}%)")
print(f"  When tumor IS present:")
tumor_nonzero = tumor_arr[tumor_arr > 0]
if len(tumor_nonzero) > 0:
    print(f"    Min coverage:    {tumor_nonzero.min():.4f}%")
    print(f"    Max coverage:    {tumor_nonzero.max():.2f}%")
    print(f"    Mean coverage:   {tumor_nonzero.mean():.4f}%")
    print(f"    Median coverage: {np.median(tumor_nonzero):.4f}%")

# ═══════════════════════════════════════════════════════════════
# GENERATE PIXEL-LEVEL PLOTS
# ═══════════════════════════════════════════════════════════════
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#161b22'
plt.rcParams['text.color'] = '#c9d1d9'
plt.rcParams['axes.labelcolor'] = '#c9d1d9'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'
plt.rcParams['axes.edgecolor'] = '#30363d'
plt.rcParams['grid.color'] = '#21262d'

# ──── PLOT 8: Pixel-level class balance (THE critical chart) ────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Pixel-Level Class Imbalance (What the Model Actually Sees)", fontsize=16, fontweight='bold', color='white')

# Pie chart
labels = [f'Background\n{100*background_pixels/total_pixels:.1f}%',
          f'Liver\n{100*liver_pixels/total_pixels:.1f}%',
          f'Tumor\n{100*tumor_pixels/total_pixels:.1f}%']
sizes = [background_pixels, liver_pixels, tumor_pixels]
colors = ['#8b949e', '#238636', '#da3633']
explode = (0, 0.05, 0.1)
wedges, texts, autotexts = axes[0].pie(sizes, labels=labels, colors=colors, explode=explode,
                                         autopct='', startangle=90,
                                         textprops={'fontsize': 10, 'color': '#c9d1d9'})
axes[0].set_title('Pixel-Level Distribution', fontsize=13, color='white')

# Liver coverage histogram (when non-zero)
if len(liver_nonzero) > 0:
    axes[1].hist(liver_nonzero, bins=50, color='#238636', edgecolor='#0d1117', alpha=0.85)
    axes[1].axvline(liver_nonzero.mean(), color='#f0883e', linestyle='--', linewidth=2,
                     label=f'Mean: {liver_nonzero.mean():.1f}%')
    axes[1].axvline(np.median(liver_nonzero), color='#1f6feb', linestyle='--', linewidth=2,
                     label=f'Median: {np.median(liver_nonzero):.1f}%')
axes[1].set_xlabel('Liver Coverage (%)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Liver Coverage per Slice (non-empty only)', fontsize=13, color='white')
axes[1].legend(fontsize=10)

# Tumor coverage histogram (when non-zero)
if len(tumor_nonzero) > 0:
    axes[2].hist(tumor_nonzero, bins=50, color='#da3633', edgecolor='#0d1117', alpha=0.85)
    axes[2].axvline(tumor_nonzero.mean(), color='#f0883e', linestyle='--', linewidth=2,
                     label=f'Mean: {tumor_nonzero.mean():.2f}%')
    axes[2].axvline(np.median(tumor_nonzero), color='#1f6feb', linestyle='--', linewidth=2,
                     label=f'Median: {np.median(tumor_nonzero):.2f}%')
axes[2].set_xlabel('Tumor Coverage (%)', fontsize=12)
axes[2].set_ylabel('Count', fontsize=12)
axes[2].set_title('Tumor Coverage per Slice (non-empty only)', fontsize=13, color='white')
axes[2].legend(fontsize=10)

plt.tight_layout()
plt.savefig('results/eda/08_pixel_level_imbalance.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
print("\n  Saved: results/eda/08_pixel_level_imbalance.png")

# ──── PLOT 9: Image intensity distribution ────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("CT Image Intensity Statistics", fontsize=16, fontweight='bold', color='white')

axes[0].hist(img_means, bins=50, color='#1f6feb', edgecolor='#0d1117', alpha=0.85)
axes[0].axvline(np.mean(img_means), color='#f0883e', linestyle='--', linewidth=2,
                 label=f'Mean: {np.mean(img_means):.3f}')
axes[0].set_xlabel('Mean Pixel Value (normalized)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Distribution of Image Mean Values', fontsize=13, color='white')
axes[0].legend(fontsize=10)

axes[1].hist(img_stds, bins=50, color='#a371f7', edgecolor='#0d1117', alpha=0.85)
axes[1].axvline(np.mean(img_stds), color='#f0883e', linestyle='--', linewidth=2,
                 label=f'Mean: {np.mean(img_stds):.3f}')
axes[1].set_xlabel('Std Dev of Pixel Values', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Distribution of Image Std Dev', fontsize=13, color='white')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('results/eda/09_image_intensity.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
print("  Saved: results/eda/09_image_intensity.png")

# ──── PLOT 10: Sample visualization (8 diverse samples) ────
print("\n  Generating sample visualizations...")

# Pick 4 liver-heavy + 4 tumor-heavy samples
liver_heavy_indices = []
tumor_heavy_indices = []
background_indices = []

for idx in sample_indices:
    row = train_df.iloc[int(idx)]
    if not row["liver_mask_empty"] and row["tumor_mask_empty"]:
        liver_heavy_indices.append(int(idx))
    elif not row["liver_mask_empty"] and not row["tumor_mask_empty"]:
        tumor_heavy_indices.append(int(idx))
    elif row["liver_mask_empty"] and row["tumor_mask_empty"]:
        background_indices.append(int(idx))

random.seed(42)
selected = []
if liver_heavy_indices:
    selected.extend(random.sample(liver_heavy_indices, min(2, len(liver_heavy_indices))))
if tumor_heavy_indices:
    selected.extend(random.sample(tumor_heavy_indices, min(4, len(tumor_heavy_indices))))
if background_indices:
    selected.extend(random.sample(background_indices, min(2, len(background_indices))))

n_show = len(selected)
if n_show > 0:
    fig, axes = plt.subplots(n_show, 4, figsize=(20, 5 * n_show))
    fig.suptitle("Sample CT Slices with Liver & Tumor Masks", fontsize=18, fontweight='bold', color='white', y=1.01)
    
    for r, idx in enumerate(selected):
        row = train_df.iloc[idx]
        img = cv2.imread(os.path.join(DATASET_DIR, os.path.basename(str(row["filepath"]))), cv2.IMREAD_GRAYSCALE)
        liv = cv2.imread(os.path.join(DATASET_DIR, os.path.basename(str(row["liver_maskpath"]))), cv2.IMREAD_GRAYSCALE)
        tum = cv2.imread(os.path.join(DATASET_DIR, os.path.basename(str(row["tumor_maskpath"]))), cv2.IMREAD_GRAYSCALE)
        
        if img is None or liv is None or tum is None:
            continue
        
        liv_bin = (liv.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        tum_bin = (tum.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        
        # Create overlay
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        overlay[liv_bin > 0] = [0, 180, 0]   # Green for liver
        overlay[tum_bin > 0] = [220, 50, 50]  # Red for tumor
        
        ax_row = axes[r] if n_show > 1 else axes
        ax_row[0].imshow(img, cmap='gray')
        ax_row[0].set_title(f'CT Slice (study={row["study_number"]})', fontsize=10, color='white')
        ax_row[1].imshow(liv_bin, cmap='Greens')
        ax_row[1].set_title(f'Liver Mask ({liv_bin.sum():.0f}px, {100*liv_bin.sum()/img.size:.1f}%)', fontsize=10, color='white')
        ax_row[2].imshow(tum_bin, cmap='Reds')
        ax_row[2].set_title(f'Tumor Mask ({tum_bin.sum():.0f}px, {100*tum_bin.sum()/img.size:.1f}%)', fontsize=10, color='white')
        ax_row[3].imshow(overlay)
        ax_row[3].set_title('Overlay (Green=Liver, Red=Tumor)', fontsize=10, color='white')
        
        for ax in ax_row:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/eda/10_sample_visualizations.png', dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    print("  Saved: results/eda/10_sample_visualizations.png")

# ──── PLOT 11: Image size analysis ────
print("\n  Checking image sizes...")
sizes = set()
for idx in sample_indices[:100]:
    row = train_df.iloc[int(idx)]
    img = cv2.imread(os.path.join(DATASET_DIR, os.path.basename(str(row["filepath"]))), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        sizes.add(img.shape)

print(f"  Unique image sizes found: {sizes}")

# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PIXEL-LEVEL EDA COMPLETE")
print("=" * 70)
print(f"""
CRITICAL FINDING — PIXEL-LEVEL IMBALANCE:
──────────────────────────────────────────
  Background: {100*background_pixels/total_pixels:.1f}% of ALL pixels
  Liver:      {100*liver_pixels/total_pixels:.2f}% of ALL pixels  
  Tumor:      {100*tumor_pixels/total_pixels:.4f}% of ALL pixels

  ★ RECOMMENDED pos_weight for BCE:
    Liver: {min(liver_pw, 50.0):.1f} (raw: {liver_pw:.1f})
    Tumor: {min(tumor_pw, 50.0):.1f} (raw: {tumor_pw:.1f})

  This is DIFFERENT from slice-level imbalance!
  → Slice-level: 67% have liver, 88% have tumor  
  → Pixel-level: liver covers only ~{100*liver_pixels/total_pixels:.1f}% of image area
  → The model sees PIXELS, not slices. Pixel-level is what matters.
""")
