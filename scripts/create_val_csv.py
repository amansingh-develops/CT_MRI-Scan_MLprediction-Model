"""
Create lits_val.csv from the 40 studies in lits_df.csv 
that are NOT in lits_train.csv or lits_test.csv.
"""
import pandas as pd
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA = r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\data'

df_all   = pd.read_csv(f'{DATA}/lits_df.csv')
df_train = pd.read_csv(f'{DATA}/lits_train.csv')
df_test  = pd.read_csv(f'{DATA}/lits_test.csv')

# Find rows in master that are NOT in train or test
train_fps = set(df_train["filepath"].values)
test_fps  = set(df_test["filepath"].values)

val_mask = ~df_all["filepath"].isin(train_fps | test_fps)
df_val = df_all[val_mask].reset_index(drop=True)

# Save
out_path = f'{DATA}/lits_val.csv'
df_val.to_csv(out_path, index=False)

# Verification
val_studies = sorted(df_val["study_number"].unique())
train_studies = set(df_train["study_number"].unique())
test_studies  = set(df_test["study_number"].unique())
val_study_set = set(df_val["study_number"].unique())

# Check no leakage
train_val_overlap = train_studies & val_study_set
test_val_overlap  = test_studies & val_study_set

print("=" * 60)
print("  lits_val.csv CREATED")
print("=" * 60)
print(f"\n  Output: {out_path}")
print(f"  Rows:    {len(df_val):,}")
print(f"  Studies: {len(val_studies)} -> {val_studies}")

print(f"\n  FINAL 3-WAY SPLIT:")
print(f"    Train: {len(df_train):,} slices ({len(train_studies)} studies)")
print(f"    Val:   {len(df_val):,} slices ({len(val_study_set)} studies)")
print(f"    Test:  {len(df_test):,} slices ({len(test_studies)} studies)")
print(f"    Total: {len(df_train)+len(df_val)+len(df_test):,} / {len(df_all):,}")

print(f"\n  DATA LEAKAGE CHECK:")
print(f"    Train-Val study overlap:  {len(train_val_overlap)} {'CLEAN' if len(train_val_overlap)==0 else 'LEAK!'}")
print(f"    Test-Val study overlap:   {len(test_val_overlap)} {'CLEAN' if len(test_val_overlap)==0 else 'LEAK!'}")

le = df_val["liver_mask_empty"].sum()
n = len(df_val)
be = ((df_val["liver_mask_empty"]) & (df_val["tumor_mask_empty"])).sum()
print(f"\n  Val empty stats:")
print(f"    Liver empty: {le:,}/{n:,} ({100*le/n:.1f}%)")
print(f"    Both empty:  {be:,}/{n:,} ({100*be/n:.1f}%)")
print(f"\n  DONE")
