import json, sys
sys.stdout.reconfigure(encoding='utf-8')

NB_PATH = r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\livertumor-model.ipynb'

# Verify it loads
nb = json.load(open(NB_PATH, encoding='utf-8'))
print(f"✅ Notebook loads successfully")
print(f"   Total cells: {len(nb['cells'])}")

for i, c in enumerate(nb['cells']):
    src = c['source']
    first_line = ''.join(src[:1]).strip()[:80] if src else 'EMPTY'
    print(f"   Cell {i:2d}: {c['cell_type']:8s} | {len(src):5d} lines | {first_line}")

# Verify Cell 12 has all fixes
cell12 = ''.join(nb['cells'][12]['source'])
fixes = {
    'GRAD_CLIP    = 5.0': False,
    'warmup_end_epoch = start_epoch + WARMUP_EPOCHS - 1': False,
    'inf_count_this_epoch': False,
    'CATASTROPHIC SPIKE': False,
    'current_epoch - start_ep': False,
    'no_improve = 0  # FIX #6': False,
}
for pattern in fixes:
    fixes[pattern] = pattern in cell12

print(f"\n{'='*50}")
print("Cell 12 fix verification:")
for pattern, found in fixes.items():
    status = '✅' if found else '❌'
    print(f"  {status} {pattern}")

# Check no duplicate keys or syntax issues
keywords = ['def set_lr', 'def get_warmup_lr', 'for epoch in range', 
            'scheduler = optim.lr_scheduler']
for kw in keywords:
    count = cell12.count(kw)
    status = '✅' if count == 1 else f'⚠️ ({count}x)'
    print(f"  {status} '{kw}' appears {count}x")

print(f"\n✅ Notebook verification complete")
