import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open(r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\livertumor-model.ipynb', encoding='utf-8'))

# Search across all code cells for key definitions
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    for kw in ['class CombinedLoss', 'def dice_score', 'class UNet']:
        if kw in src:
            idx = src.find(kw)
            end = min(idx + 600, len(src))
            print(f"Cell {i}: {kw}")
            print(src[idx:end])
            print("...\n")
