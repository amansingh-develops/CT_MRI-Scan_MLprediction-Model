import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB = r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\livertumor-model.ipynb'
with open(NB, encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"Total cells: {len(nb['cells'])}")
print(f"Code cells:  {len(code_cells)}")

for i, c in enumerate(code_cells):
    src = c['source'][:100].replace('\n', ' ')
    print(f"  Code {i}: {src}")
