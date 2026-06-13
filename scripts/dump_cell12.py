import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open(r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\livertumor-model.ipynb', encoding='utf-8'))

# Search across ALL cells for tqdm import
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'import' in src and 'tqdm' in src:
        lines = src.split('\n')
        for j, line in enumerate(lines):
            if 'tqdm' in line:
                print(f"Cell {i}, line {j+1}: {line.strip()}")
