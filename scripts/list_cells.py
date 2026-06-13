import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open(r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\livertumor-model.ipynb', encoding='utf-8'))

for i, c in enumerate(nb['cells']):
    src = c['source']
    first_line = ''.join(src[:1]).strip()[:100] if src else 'EMPTY'
    print(f'Cell {i}: {c["cell_type"]}, {len(src)} lines, first={first_line}')
