import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open(r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\livertumor-model.ipynb', encoding='utf-8'))

cell = ''.join(nb['cells'][12]['source'])
# Print from 5000 to 7000
print(cell[5000:7000])
