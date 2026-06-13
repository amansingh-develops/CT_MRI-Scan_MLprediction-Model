import pandas as pd
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA = r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\data'

df_all   = pd.read_csv(f'{DATA}/lits_df.csv')
df_train = pd.read_csv(f'{DATA}/lits_train.csv')
df_test  = pd.read_csv(f'{DATA}/lits_test.csv')
df_probe = pd.read_csv(f'{DATA}/lits_probe.csv')

print("=" * 60)
print("  CSV FILE ANALYSIS")
print("=" * 60)

print(f"\n  lits_df.csv (FULL):    {len(df_all):,} rows")
print(f"  lits_train.csv:        {len(df_train):,} rows")
print(f"  lits_test.csv:         {len(df_test):,} rows")
print(f"  lits_probe.csv:        {len(df_probe):,} rows")
print(f"  train + test         = {len(df_train)+len(df_test):,}")

# Are test and probe identical?
print(f"\n  test == probe? {df_test.equals(df_probe)}")

# Study numbers
train_studies = sorted(df_train["study_number"].unique())
test_studies  = sorted(df_test["study_number"].unique())
all_studies   = sorted(df_all["study_number"].unique())
probe_studies = sorted(df_probe["study_number"].unique())

print(f"\n  All studies ({len(all_studies)}):   {all_studies}")
print(f"  Train studies ({len(train_studies)}): {train_studies}")
print(f"  Test studies ({len(test_studies)}):  {test_studies}")
print(f"  Probe studies ({len(probe_studies)}): {probe_studies}")

# Overlap analysis
train_fps = set(df_train["filepath"].values)
test_fps  = set(df_test["filepath"].values)
all_fps   = set(df_all["filepath"].values)

overlap = train_fps & test_fps
print(f"\n  Train-Test overlap:  {len(overlap)} files")

remaining = all_fps - train_fps - test_fps
print(f"  In lits_df but NOT in train or test: {len(remaining)} files")

if len(remaining) > 0:
    rem_df = df_all[df_all["filepath"].isin(remaining)]
    rem_studies = sorted(rem_df["study_number"].unique())
    print(f"  Remaining studies: {rem_studies}")

# Check: is lits_df = train + test?
full_minus_train = all_fps - train_fps
print(f"\n  lits_df - train = {len(full_minus_train)} rows")
print(f"  lits_test rows  = {len(test_fps)}")
print(f"  Match? {full_minus_train == test_fps}")

# Check what Kaggle uses as val
print("\n" + "=" * 60)
print("  WHAT THE v8 NOTEBOOK USES")
print("=" * 60)
print("  Currently: TRAIN_CSV = lits_train.csv")
print("  Currently: VAL_CSV   = lits_val.csv (from Kaggle dataset)")
print("  Your local data has: lits_train + lits_test + lits_probe + lits_df")
print("  There is NO separate lits_val.csv in your local data/")

# Empty slice stats
print("\n" + "=" * 60)
print("  EMPTY SLICE STATS")
print("=" * 60)
for name, d in [("full", df_all), ("train", df_train), ("test", df_test)]:
    n = len(d)
    le = d["liver_mask_empty"].sum()
    te = d["tumor_mask_empty"].sum()
    be = ((d["liver_mask_empty"]) & (d["tumor_mask_empty"])).sum()
    print(f"  {name:6s}: {n:,} total | liver_empty={le:,} ({100*le/n:.1f}%) | both_empty={be:,} ({100*be/n:.1f}%)")

# Columns comparison
print("\n" + "=" * 60)
print("  COLUMN COMPARISON")
print("=" * 60)
print(f"  lits_df:    {list(df_all.columns)}")
print(f"  lits_train: {list(df_train.columns)}")
print(f"  lits_test:  {list(df_test.columns)}")
print(f"  lits_probe: {list(df_probe.columns)}")
