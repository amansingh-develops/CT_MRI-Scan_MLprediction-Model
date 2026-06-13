import pandas as pd
import os

def analyze_csv(filepath, name):
    print(f"\n{'='*50}")
    print(f"Analyzing {name}: {filepath}")
    print(f"{'='*50}")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    df = pd.read_csv(filepath)
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing/Null values per column:")
    print(df.isnull().sum())
    
    if 'liver_mask_empty' in df.columns and 'tumor_mask_empty' in df.columns:
        total = len(df)
        liver_present = len(df[df['liver_mask_empty'] == False])
        tumor_present = len(df[df['tumor_mask_empty'] == False])
        both_present = len(df[(df['liver_mask_empty'] == False) & (df['tumor_mask_empty'] == False)])
        neither_present = len(df[(df['liver_mask_empty'] == True) & (df['tumor_mask_empty'] == True)])
        
        print("\nClass Balance:")
        print(f"Total slices:   {total}")
        print(f"Liver present:  {liver_present} ({(liver_present/total)*100:.1f}%)")
        print(f"Tumor present:  {tumor_present} ({(tumor_present/total)*100:.1f}%)")
        print(f"Both present:   {both_present} ({(both_present/total)*100:.1f}%)")
        print(f"Neither:        {neither_present} ({(neither_present/total)*100:.1f}%)")
        
    if 'study_number' in df.columns:
        print("\nPer-patient slice count (Top 10 studies):")
        print(df['study_number'].value_counts().head(10))

if __name__ == "__main__":
    analyze_csv("data/lits_train.csv", "Train Dataset")
    analyze_csv("data/lits_test.csv", "Test Dataset")
    analyze_csv("data/lits_probe.csv", "Probe Dataset")
    analyze_csv("data/lits_df.csv", "Full DataFrame (lits_df)")
