import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
Comprehensive EDA for LiTS Liver Tumor Segmentation Dataset
=============================================================
Analyzes all 4 CSV files, computes class distributions, checks data quality,
and generates publication-quality visualizations.

Output: results/eda/ directory with all plots + printed analysis
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

# ═══════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════
os.makedirs("results/eda", exist_ok=True)

DATA_DIR = "data"
csv_files = {
    "lits_df":    os.path.join(DATA_DIR, "lits_df.csv"),
    "lits_train": os.path.join(DATA_DIR, "lits_train.csv"),
    "lits_probe": os.path.join(DATA_DIR, "lits_probe.csv"),  # validation
    "lits_test":  os.path.join(DATA_DIR, "lits_test.csv"),
}

# Load all CSVs
dfs = {}
for name, path in csv_files.items():
    if os.path.exists(path):
        dfs[name] = pd.read_csv(path)
        print(f"Loaded {name}: {dfs[name].shape[0]:,} rows, {dfs[name].shape[1]} columns")
    else:
        print(f"NOT FOUND: {path}")

print("\n" + "=" * 70)
print("  SECTION 1: DATASET OVERVIEW")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# 1. BASIC STATISTICS
# ═══════════════════════════════════════════════════════════════
full_df = dfs["lits_df"]
train_df = dfs["lits_train"]
val_df = dfs["lits_probe"]
test_df = dfs["lits_test"]

print(f"\nTotal samples (lits_df):     {len(full_df):,}")
print(f"Training samples:            {len(train_df):,}")
print(f"Validation samples (probe):  {len(val_df):,}")
print(f"Test samples:                {len(test_df):,}")
print(f"Sum (train+val+test):        {len(train_df)+len(val_df)+len(test_df):,}")
print(f"Overlap check (sum vs full): {len(full_df) - (len(train_df)+len(val_df)+len(test_df)):+,} difference")

# Split ratios
total = len(train_df) + len(val_df) + len(test_df)
print(f"\nSplit ratios:")
print(f"  Train: {len(train_df)/total*100:.1f}%")
print(f"  Val:   {len(val_df)/total*100:.1f}%")
print(f"  Test:  {len(test_df)/total*100:.1f}%")

# ═══════════════════════════════════════════════════════════════
# 2. CLASS DISTRIBUTION (Critical for understanding imbalance)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 2: CLASS DISTRIBUTION (LIVER vs TUMOR)")
print("=" * 70)

for name, df in [("Full Dataset", full_df), ("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    n = len(df)
    liver_nonempty = (~df["liver_mask_empty"]).sum()
    tumor_nonempty = (~df["tumor_mask_empty"]).sum()
    both_empty = ((df["liver_mask_empty"]) & (df["tumor_mask_empty"])).sum()
    liver_only = ((~df["liver_mask_empty"]) & (df["tumor_mask_empty"])).sum()
    both_present = ((~df["liver_mask_empty"]) & (~df["tumor_mask_empty"])).sum()
    tumor_only = ((df["liver_mask_empty"]) & (~df["tumor_mask_empty"])).sum()
    
    print(f"\n--- {name} ({n:,} slices) ---")
    print(f"  Liver non-empty:  {liver_nonempty:>6,} ({100*liver_nonempty/n:5.1f}%)")
    print(f"  Tumor non-empty:  {tumor_nonempty:>6,} ({100*tumor_nonempty/n:5.1f}%)")
    print(f"  Both empty:       {both_empty:>6,} ({100*both_empty/n:5.1f}%)  <- pure background")
    print(f"  Liver only:       {liver_only:>6,} ({100*liver_only/n:5.1f}%)  <- liver visible, no tumor")
    print(f"  Both present:     {both_present:>6,} ({100*both_present/n:5.1f}%)  <- liver + tumor visible")
    print(f"  Tumor only:       {tumor_only:>6,} ({100*tumor_only/n:5.1f}%)  <- should be rare/0")

# ═══════════════════════════════════════════════════════════════
# 3. STUDY (PATIENT/VOLUME) ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 3: STUDY (PATIENT/VOLUME) ANALYSIS")
print("=" * 70)

n_studies_full = full_df["study_number"].nunique()
print(f"\nTotal unique studies (patients/volumes): {n_studies_full}")

# Slices per study
slices_per_study = full_df.groupby("study_number").size()
print(f"\nSlices per study:")
print(f"  Min:    {slices_per_study.min()}")
print(f"  Max:    {slices_per_study.max()}")
print(f"  Mean:   {slices_per_study.mean():.1f}")
print(f"  Median: {slices_per_study.median():.1f}")
print(f"  Std:    {slices_per_study.std():.1f}")

# Studies with liver/tumor content
study_liver = full_df.groupby("study_number")["liver_mask_empty"].apply(lambda x: (~x).any())
study_tumor = full_df.groupby("study_number")["tumor_mask_empty"].apply(lambda x: (~x).any())
print(f"\nStudies with liver content: {study_liver.sum()}/{n_studies_full} ({100*study_liver.sum()/n_studies_full:.1f}%)")
print(f"Studies with tumor content: {study_tumor.sum()}/{n_studies_full} ({100*study_tumor.sum()/n_studies_full:.1f}%)")

# Liver slices per study (for studies that have liver)
liver_slices_per_study = full_df[~full_df["liver_mask_empty"]].groupby("study_number").size()
print(f"\nLiver slices per study (among studies with liver):")
print(f"  Min:    {liver_slices_per_study.min()}")
print(f"  Max:    {liver_slices_per_study.max()}")
print(f"  Mean:   {liver_slices_per_study.mean():.1f}")
print(f"  Median: {liver_slices_per_study.median():.1f}")

# Tumor slices per study (for studies that have tumor)
tumor_slices = full_df[~full_df["tumor_mask_empty"]]
if len(tumor_slices) > 0:
    tumor_slices_per_study = tumor_slices.groupby("study_number").size()
    print(f"\nTumor slices per study (among studies with tumor):")
    print(f"  Min:    {tumor_slices_per_study.min()}")
    print(f"  Max:    {tumor_slices_per_study.max()}")
    print(f"  Mean:   {tumor_slices_per_study.mean():.1f}")
    print(f"  Median: {tumor_slices_per_study.median():.1f}")

# ═══════════════════════════════════════════════════════════════
# 4. TRAIN/VAL/TEST STUDY OVERLAP CHECK
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 4: DATA LEAKAGE CHECK (Study Overlap)")
print("=" * 70)

train_studies = set(train_df["study_number"].unique())
val_studies = set(val_df["study_number"].unique())
test_studies = set(test_df["study_number"].unique())

print(f"\nUnique studies - Train: {len(train_studies)}, Val: {len(val_studies)}, Test: {len(test_studies)}")

train_val_overlap = train_studies & val_studies
train_test_overlap = train_studies & test_studies
val_test_overlap = val_studies & test_studies

if train_val_overlap:
    print(f"  ⚠️ OVERLAP Train-Val: {len(train_val_overlap)} studies: {sorted(train_val_overlap)[:10]}...")
else:
    print(f"  ✅ No Train-Val overlap")

if train_test_overlap:
    print(f"  ⚠️ OVERLAP Train-Test: {len(train_test_overlap)} studies: {sorted(train_test_overlap)[:10]}...")
else:
    print(f"  ✅ No Train-Test overlap")

if val_test_overlap:
    print(f"  ⚠️ OVERLAP Val-Test: {len(val_test_overlap)} studies")
else:
    print(f"  ✅ No Val-Test overlap")

# ═══════════════════════════════════════════════════════════════
# 5. INSTANCE NUMBER ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 5: INSTANCE NUMBER (SLICE INDEX) ANALYSIS")
print("=" * 70)

print(f"\nInstance number range: {full_df['instance_number'].min()} to {full_df['instance_number'].max()}")
print(f"Unique instance numbers: {full_df['instance_number'].nunique()}")

# Check if instance_number is always 0 (possible issue)
if full_df['instance_number'].nunique() == 1:
    print(f"  ⚠️ WARNING: instance_number is always {full_df['instance_number'].iloc[0]}")
    print(f"  This likely means individual slice IDs are encoded in the filename, not here.")
else:
    print(f"  Instance number distribution:")
    print(full_df['instance_number'].describe())

# ═══════════════════════════════════════════════════════════════
# 6. FILENAME PATTERN ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 6: FILENAME PATTERN ANALYSIS")
print("=" * 70)

# Extract slice numbers from filenames
import re

def extract_slice_num(filepath):
    """Extract slice number from filename like volume-0_123.png"""
    match = re.search(r'volume-\d+_(\d+)\.png', str(filepath))
    if match:
        return int(match.group(1))
    return None

def extract_vol_num(filepath):
    """Extract volume number from filename like volume-0_123.png"""
    match = re.search(r'volume-(\d+)_\d+\.png', str(filepath))
    if match:
        return int(match.group(1))
    return None

full_df_copy = full_df.copy()
full_df_copy["slice_num"] = full_df_copy["filepath"].apply(extract_slice_num)
full_df_copy["vol_num"] = full_df_copy["filepath"].apply(extract_vol_num)

print(f"\nExtracted volume numbers: {sorted(full_df_copy['vol_num'].dropna().unique().astype(int))[:20]}...")
print(f"Volume count: {full_df_copy['vol_num'].nunique()}")
print(f"\nSlice number range: {full_df_copy['slice_num'].min()} to {full_df_copy['slice_num'].max()}")

# Check: does volume_num == study_number?
if full_df_copy['vol_num'].notna().all():
    match_check = (full_df_copy['vol_num'].astype(int) == full_df_copy['study_number']).mean()
    print(f"\nVolume_num == study_number match rate: {match_check*100:.1f}%")

# ═══════════════════════════════════════════════════════════════
# 7. FILE EXISTENCE CHECK (sample)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 7: LOCAL FILE EXISTENCE CHECK")
print("=" * 70)

dataset_dir = os.path.join(DATA_DIR, "dataset_6")
if os.path.exists(dataset_dir):
    local_files = os.listdir(dataset_dir)
    print(f"\nFiles in dataset_6/: {len(local_files)}")
    
    # Categorize files
    volumes = [f for f in local_files if f.startswith("volume-")]
    liver_masks = [f for f in local_files if "livermask" in f]
    tumor_masks = [f for f in local_files if "lesionmask" in f]
    
    print(f"  Volume images:  {len(volumes)}")
    print(f"  Liver masks:    {len(liver_masks)}")
    print(f"  Tumor masks:    {len(tumor_masks)}")
    
    if len(volumes) > 0:
        print(f"\n  Sample volume:     {volumes[0]}")
    if len(liver_masks) > 0:
        print(f"  Sample liver mask: {liver_masks[0]}")
    if len(tumor_masks) > 0:
        print(f"  Sample tumor mask: {tumor_masks[0]}")
    
    # Check if CSV references match local files
    sample_csv_files = train_df["filepath"].apply(lambda x: os.path.basename(str(x))).head(20)
    found = sum(1 for f in sample_csv_files if f in local_files)
    print(f"\n  CSV-to-local match (first 20): {found}/20")
else:
    print(f"\n  dataset_6/ not found locally — data is on Kaggle only")
    print(f"  (This is expected — images are too large for local storage)")

# ═══════════════════════════════════════════════════════════════
# GENERATE VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  GENERATING EDA VISUALIZATIONS")
print("=" * 70)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#161b22'
plt.rcParams['text.color'] = '#c9d1d9'
plt.rcParams['axes.labelcolor'] = '#c9d1d9'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'
plt.rcParams['axes.edgecolor'] = '#30363d'
plt.rcParams['grid.color'] = '#21262d'
plt.rcParams['font.family'] = 'sans-serif'

# ──────────────────────────────────────────────────────────────
# PLOT 1: Class Distribution Overview (4 subplots)
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("LiTS Dataset — Class Distribution Analysis", fontsize=18, fontweight='bold', color='white', y=0.98)

datasets_to_plot = [("Full Dataset", full_df), ("Train", train_df), ("Validation", val_df), ("Test", test_df)]
colors_pie = ['#238636', '#1f6feb', '#f0883e', '#8b949e']

for ax, (name, df) in zip(axes.flat, datasets_to_plot):
    both_empty = ((df["liver_mask_empty"]) & (df["tumor_mask_empty"])).sum()
    liver_only = ((~df["liver_mask_empty"]) & (df["tumor_mask_empty"])).sum()
    both_present = ((~df["liver_mask_empty"]) & (~df["tumor_mask_empty"])).sum()
    tumor_only = ((df["liver_mask_empty"]) & (~df["tumor_mask_empty"])).sum()
    
    sizes = [both_empty, liver_only, both_present, tumor_only]
    labels = [
        f'Background\n{both_empty:,} ({100*both_empty/len(df):.1f}%)',
        f'Liver Only\n{liver_only:,} ({100*liver_only/len(df):.1f}%)',
        f'Liver+Tumor\n{both_present:,} ({100*both_present/len(df):.1f}%)',
        f'Tumor Only\n{tumor_only:,} ({100*tumor_only/len(df):.1f}%)'
    ]
    
    # Only include non-zero segments
    nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors_pie) if s > 0]
    if nonzero:
        sz, lb, cl = zip(*nonzero)
        wedges, texts, autotexts = ax.pie(sz, labels=lb, colors=cl, autopct='', startangle=90,
                                           textprops={'fontsize': 8, 'color': '#c9d1d9'})
    ax.set_title(f'{name} ({len(df):,} slices)', fontsize=13, color='white', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('results/eda/01_class_distribution.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
print("  Saved: results/eda/01_class_distribution.png")

# ──────────────────────────────────────────────────────────────
# PLOT 2: Slices per Study (Histogram)
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Slices per Study (Patient Volume)", fontsize=16, fontweight='bold', color='white')

axes[0].hist(slices_per_study.values, bins=30, color='#1f6feb', edgecolor='#0d1117', alpha=0.85)
axes[0].set_xlabel('Number of Slices', fontsize=12)
axes[0].set_ylabel('Number of Studies', fontsize=12)
axes[0].set_title('Distribution of Slices per Study', fontsize=13, color='white')
axes[0].axvline(slices_per_study.mean(), color='#f0883e', linestyle='--', linewidth=2, label=f'Mean: {slices_per_study.mean():.0f}')
axes[0].axvline(slices_per_study.median(), color='#238636', linestyle='--', linewidth=2, label=f'Median: {slices_per_study.median():.0f}')
axes[0].legend(fontsize=10)

# Liver content proportion per study
liver_prop = full_df.groupby("study_number").apply(lambda x: (~x["liver_mask_empty"]).mean())
axes[1].hist(liver_prop.values, bins=30, color='#da3633', edgecolor='#0d1117', alpha=0.85)
axes[1].set_xlabel('Proportion of Slices with Liver', fontsize=12)
axes[1].set_ylabel('Number of Studies', fontsize=12)
axes[1].set_title('Liver Content Ratio per Study', fontsize=13, color='white')
axes[1].axvline(liver_prop.mean(), color='#f0883e', linestyle='--', linewidth=2, label=f'Mean: {liver_prop.mean():.2f}')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('results/eda/02_slices_per_study.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
print("  Saved: results/eda/02_slices_per_study.png")

# ──────────────────────────────────────────────────────────────
# PLOT 3: Train/Val/Test Split Visualization
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Train / Validation / Test Split Analysis", fontsize=16, fontweight='bold', color='white')

# Bar chart: total slices
split_names = ['Train', 'Validation', 'Test']
split_sizes = [len(train_df), len(val_df), len(test_df)]
split_colors = ['#238636', '#1f6feb', '#f0883e']
bars = axes[0].bar(split_names, split_sizes, color=split_colors, edgecolor='#0d1117', width=0.6)
for bar, size in zip(bars, split_sizes):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{size:,}', ha='center', va='bottom', fontsize=12, color='white', fontweight='bold')
axes[0].set_ylabel('Number of Slices', fontsize=12)
axes[0].set_title('Total Slices per Split', fontsize=13, color='white')

# Stacked bar: liver/tumor composition per split
for idx, (name, df, color) in enumerate(zip(split_names, [train_df, val_df, test_df], split_colors)):
    bg = ((df["liver_mask_empty"]) & (df["tumor_mask_empty"])).sum()
    liver = ((~df["liver_mask_empty"]) & (df["tumor_mask_empty"])).sum()
    both = ((~df["liver_mask_empty"]) & (~df["tumor_mask_empty"])).sum()
    
    total = len(df)
    axes[1].bar(idx, 100*bg/total, color='#8b949e', edgecolor='#0d1117', width=0.6, label='Background' if idx == 0 else '')
    axes[1].bar(idx, 100*liver/total, bottom=100*bg/total, color='#238636', edgecolor='#0d1117', width=0.6, label='Liver Only' if idx == 0 else '')
    axes[1].bar(idx, 100*both/total, bottom=100*(bg+liver)/total, color='#da3633', edgecolor='#0d1117', width=0.6, label='Liver+Tumor' if idx == 0 else '')

axes[1].set_xticks(range(3))
axes[1].set_xticklabels(split_names)
axes[1].set_ylabel('Percentage (%)', fontsize=12)
axes[1].set_title('Composition per Split', fontsize=13, color='white')
axes[1].legend(fontsize=10, loc='upper right')
axes[1].set_ylim(0, 105)

plt.tight_layout()
plt.savefig('results/eda/03_train_val_test_split.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
print("  Saved: results/eda/03_train_val_test_split.png")

# ──────────────────────────────────────────────────────────────
# PLOT 4: Per-Study Liver/Tumor Content Heatmap
# ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 6))
fig.suptitle("Liver & Tumor Content by Study (Volume)", fontsize=16, fontweight='bold', color='white')

# For each study, compute: liver proportion, tumor proportion
study_stats = full_df.groupby("study_number").agg(
    total_slices=("filepath", "count"),
    liver_count=("liver_mask_empty", lambda x: (~x).sum()),
    tumor_count=("tumor_mask_empty", lambda x: (~x).sum())
).reset_index()
study_stats = study_stats.sort_values("study_number")

x = np.arange(len(study_stats))
width = 0.35

bars1 = ax.bar(x - width/2, study_stats["liver_count"], width, label='Liver Slices', color='#238636', alpha=0.85)
bars2 = ax.bar(x + width/2, study_stats["tumor_count"], width, label='Tumor Slices', color='#da3633', alpha=0.85)
ax.plot(x, study_stats["total_slices"], 'o-', color='#f0883e', markersize=3, linewidth=1.5, label='Total Slices', alpha=0.7)

ax.set_xlabel('Study Number', fontsize=12)
ax.set_ylabel('Number of Slices', fontsize=12)
ax.set_title('Content Distribution Across Studies', fontsize=13, color='white')
ax.legend(fontsize=10)

# Show every 5th study label
tick_indices = list(range(0, len(study_stats), max(1, len(study_stats)//20)))
ax.set_xticks([x[i] for i in tick_indices])
ax.set_xticklabels([study_stats.iloc[i]["study_number"] for i in tick_indices], fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig('results/eda/04_per_study_content.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
print("  Saved: results/eda/04_per_study_content.png")

# ──────────────────────────────────────────────────────────────
# PLOT 5: Imbalance summary — THE critical chart
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Class Imbalance Analysis — THE Root Cause of Model Failure", fontsize=16, fontweight='bold', color='white')

# Slice-level imbalance
liver_present = (~full_df["liver_mask_empty"]).sum()
liver_absent = full_df["liver_mask_empty"].sum()
tumor_present = (~full_df["tumor_mask_empty"]).sum()
tumor_absent = full_df["tumor_mask_empty"].sum()

categories = ['Has Liver\nMask', 'Empty Liver\nMask', 'Has Tumor\nMask', 'Empty Tumor\nMask']
values = [liver_present, liver_absent, tumor_present, tumor_absent]
colors = ['#238636', '#8b949e', '#da3633', '#8b949e']

bars = axes[0].bar(categories, values, color=colors, edgecolor='#0d1117', width=0.6)
for bar, val in zip(bars, values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{val:,}', ha='center', va='bottom', fontsize=10, color='white')
axes[0].set_ylabel('Number of Slices', fontsize=12)
axes[0].set_title('Slice-Level Class Counts', fontsize=13, color='white')

# Imbalance ratio visualization
liver_ratio = liver_absent / max(liver_present, 1)
tumor_ratio = tumor_absent / max(tumor_present, 1)

ratio_names = ['Liver\n(neg:pos)', 'Tumor\n(neg:pos)']
ratio_values = [liver_ratio, tumor_ratio]
ratio_colors = ['#238636', '#da3633']

bars = axes[1].bar(ratio_names, ratio_values, color=ratio_colors, edgecolor='#0d1117', width=0.4)
for bar, val in zip(bars, ratio_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}:1', ha='center', va='bottom', fontsize=14, color='white', fontweight='bold')
axes[1].set_ylabel('Negative:Positive Ratio', fontsize=12)
axes[1].set_title('Class Imbalance Ratio (Higher = More Imbalanced)', fontsize=13, color='white')
axes[1].axhline(y=1.0, color='#f0883e', linestyle='--', linewidth=1.5, alpha=0.5, label='Balanced (1:1)')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('results/eda/05_class_imbalance.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
print("  Saved: results/eda/05_class_imbalance.png")

# ──────────────────────────────────────────────────────────────
# PLOT 6: Spatial distribution — where in the volume is liver/tumor?
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Spatial Distribution: Where in the Volume are Liver & Tumor?", fontsize=16, fontweight='bold', color='white')

# For each study, normalize slice position to [0, 1]
full_df_copy["slice_position_norm"] = full_df_copy.groupby("study_number")["slice_num"].transform(
    lambda x: (x - x.min()) / max(x.max() - x.min(), 1)
)

# Liver spatial distribution
liver_slices = full_df_copy[~full_df_copy["liver_mask_empty"]]
axes[0].hist(liver_slices["slice_position_norm"].dropna(), bins=50, color='#238636', 
             edgecolor='#0d1117', alpha=0.85, density=True)
axes[0].set_xlabel('Normalized Slice Position (0=bottom, 1=top)', fontsize=11)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title(f'Liver Appearance ({len(liver_slices):,} slices)', fontsize=13, color='white')

# Tumor spatial distribution
tumor_slices_df = full_df_copy[~full_df_copy["tumor_mask_empty"]]
if len(tumor_slices_df) > 0:
    axes[1].hist(tumor_slices_df["slice_position_norm"].dropna(), bins=50, color='#da3633', 
                 edgecolor='#0d1117', alpha=0.85, density=True)
axes[1].set_xlabel('Normalized Slice Position (0=bottom, 1=top)', fontsize=11)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title(f'Tumor Appearance ({len(tumor_slices_df):,} slices)', fontsize=13, color='white')

plt.tight_layout()
plt.savefig('results/eda/06_spatial_distribution.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
print("  Saved: results/eda/06_spatial_distribution.png")

# ══════════════════════════════════════════════════════════════════
# PLOT 7: Summary Statistics Dashboard
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('#0d1117')
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

fig.suptitle("LiTS Dataset — EDA Summary Dashboard", fontsize=20, fontweight='bold', color='white', y=0.98)

# Key metrics as text boxes
def add_metric_box(ax, title, value, subtitle="", color='#1f6feb'):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, facecolor='#161b22', 
                                edgecolor=color, linewidth=2, transform=ax.transAxes))
    ax.text(0.5, 0.75, title, ha='center', va='center', fontsize=10, color='#8b949e', transform=ax.transAxes)
    ax.text(0.5, 0.45, str(value), ha='center', va='center', fontsize=22, color=color, 
            fontweight='bold', transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.18, subtitle, ha='center', va='center', fontsize=9, color='#8b949e', transform=ax.transAxes)

# Row 1: Key numbers
ax1 = fig.add_subplot(gs[0, 0])
add_metric_box(ax1, "Total Slices", f"{len(full_df):,}", "across all studies", '#1f6feb')

ax2 = fig.add_subplot(gs[0, 1])
add_metric_box(ax2, "Total Studies", f"{n_studies_full}", "unique patient volumes", '#1f6feb')

ax3 = fig.add_subplot(gs[0, 2])
liver_pct = 100 * liver_present / len(full_df)
add_metric_box(ax3, "Liver Slices", f"{liver_pct:.1f}%", f"{liver_present:,} of {len(full_df):,}", '#238636')

ax4 = fig.add_subplot(gs[0, 3])
tumor_pct = 100 * tumor_present / len(full_df)
add_metric_box(ax4, "Tumor Slices", f"{tumor_pct:.1f}%", f"{tumor_present:,} of {len(full_df):,}", '#da3633')

# Row 2: Imbalance ratios
ax5 = fig.add_subplot(gs[1, 0])
add_metric_box(ax5, "Liver Imbalance", f"{liver_ratio:.1f}:1", "neg:pos ratio", '#f0883e')

ax6 = fig.add_subplot(gs[1, 1])
add_metric_box(ax6, "Tumor Imbalance", f"{tumor_ratio:.1f}:1", "neg:pos ratio", '#f0883e')

ax7 = fig.add_subplot(gs[1, 2])
add_metric_box(ax7, "Train Size", f"{len(train_df):,}", f"{100*len(train_df)/total:.0f}% of total", '#238636')

ax8 = fig.add_subplot(gs[1, 3])
overlap_status = "NONE" if not (train_val_overlap or train_test_overlap) else "LEAK!"
leak_color = '#238636' if overlap_status == "NONE" else '#da3633'
add_metric_box(ax8, "Data Leakage", overlap_status, "train/val/test overlap", leak_color)

# Row 3: Slices per study box plot + liver proportion
ax9 = fig.add_subplot(gs[2, :2])
bp = ax9.boxplot(slices_per_study.values, vert=False, patch_artist=True,
                  boxprops=dict(facecolor='#1f6feb', edgecolor='#c9d1d9'),
                  medianprops=dict(color='#f0883e', linewidth=2),
                  whiskerprops=dict(color='#c9d1d9'),
                  capprops=dict(color='#c9d1d9'),
                  flierprops=dict(markerfacecolor='#da3633', marker='o', markersize=4))
ax9.set_xlabel('Slices per Study', fontsize=12)
ax9.set_title('Slices per Study Distribution', fontsize=13, color='white')

ax10 = fig.add_subplot(gs[2, 2:])
train_liver_pct = 100 * (~train_df["liver_mask_empty"]).mean()
val_liver_pct = 100 * (~val_df["liver_mask_empty"]).mean()
test_liver_pct = 100 * (~test_df["liver_mask_empty"]).mean()
bars = ax10.bar(['Train', 'Val', 'Test'], [train_liver_pct, val_liver_pct, test_liver_pct],
                color=['#238636', '#1f6feb', '#f0883e'], edgecolor='#0d1117', width=0.5)
for bar, pct in zip(bars, [train_liver_pct, val_liver_pct, test_liver_pct]):
    ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
              f'{pct:.1f}%', ha='center', fontsize=11, color='white', fontweight='bold')
ax10.set_ylabel('% Liver Slices', fontsize=12)
ax10.set_title('Liver Content % per Split', fontsize=13, color='white')

plt.savefig('results/eda/07_summary_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
print("  Saved: results/eda/07_summary_dashboard.png")

# ═══════════════════════════════════════════════════════════════
# FINAL ANALYSIS & RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 8: KEY FINDINGS & RECOMMENDATIONS")
print("=" * 70)

print(f"""
KEY FINDINGS:
─────────────
1. DATASET SIZE: {len(full_df):,} total 2D slices from {n_studies_full} CT volumes
   → Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}

2. SEVERE CLASS IMBALANCE (ROOT CAUSE OF MODEL FAILURE):
   → Only {liver_pct:.1f}% of slices contain liver ({liver_present:,}/{len(full_df):,})
   → Only {tumor_pct:.1f}% of slices contain tumor ({tumor_present:,}/{len(full_df):,})
   → Liver neg:pos ratio = {liver_ratio:.1f}:1
   → Tumor neg:pos ratio = {tumor_ratio:.1f}:1
   ★ This explains why the model collapsed to all-zeros: predicting nothing
     gives ~{100*(1-liver_pct/100):.0f}% accuracy (background is dominant class)

3. DATA LEAKAGE: {'CLEAN — No study overlap between splits' if not (train_val_overlap or train_test_overlap) else 'WARNING — Study overlap detected!'}

4. SPATIAL DISTRIBUTION: Liver and tumor appear in specific regions
   of the CT volume, not uniformly distributed.

5. VARIABLE VOLUME SIZES: Studies range from {slices_per_study.min()} to {slices_per_study.max()} slices
   (mean: {slices_per_study.mean():.0f}, median: {slices_per_study.median():.0f})

RECOMMENDATIONS FOR MODEL TRAINING:
────────────────────────────────────
1. ★ pos_weight in BCE = crucial (already implemented in v2 notebook)
   → Liver: ~{min(liver_ratio, 50):.0f}x weight
   → Tumor: ~{min(tumor_ratio, 50):.0f}x weight (capped at 50x for stability)

2. Consider FILTERING empty slices from training:
   → Currently {100*both_empty/(len(full_df)):.0f}% of slices are pure background
   → Training on 100% of them wastes GPU time on uninformative samples
   → Option: Keep only 20-30% of empty slices, all liver/tumor slices

3. Use per-sample Dice metric (already implemented in v2)

4. Monitor training with batch diagnostics (already implemented in v2)

5. For future improvement:
   → Attention mechanisms (focus on liver region)
   → Deep supervision (auxiliary losses at decoder stages)
   → Focal Loss instead of weighted BCE (handles imbalance differently)
""")

print("=" * 70)
print("  EDA COMPLETE — All plots saved to results/eda/")
print("=" * 70)
