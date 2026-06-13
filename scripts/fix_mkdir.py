import json

with open('livertumor-model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix Cell 8 (diagnostic cell) — add os.makedirs before savefig
cell8 = nb['cells'][8]['source']

old = "plt.savefig('results/data_diagnostic.png'"
new = "os.makedirs('results', exist_ok=True)\n    plt.savefig('results/data_diagnostic.png'"

cell8 = cell8.replace(old, new)
nb['cells'][8]['source'] = cell8

# Verify the fix was applied
assert "os.makedirs('results', exist_ok=True)" in nb['cells'][8]['source'], "Fix not applied!"

with open('livertumor-model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Fixed! Added os.makedirs('results') before savefig in diagnostic cell.")
