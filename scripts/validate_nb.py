import json, os

path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "livertumor-model.ipynb")

with open(path, encoding='utf-8') as f:
    nb = json.load(f)

print(f"Valid JSON: YES")
print(f"Cells: {len(nb['cells'])}")
print(f"nbformat: {nb['nbformat']}")
for i, c in enumerate(nb['cells']):
    src = c['source'][:80].replace('\n', ' ').strip()
    print(f"  Cell {i:2d}: {c['cell_type']:8s} | {src}")

# Check critical v8 patterns
full_src = json.dumps(nb)

checks = [
    (".float()", "FP32 Dice loss"),
    ("ACCUM_STEPS", "Gradient accumulation"),
    ("WeightedRandomSampler", "Weighted sampling"),
    ("BATCH         = 16", "Batch size 16"),
    ("expandable_segments", "CUDA alloc config"),
    ("3e-4", "LR 3e-4 from scratch"),
    ("1e-5", "LR 1e-5 for fine-tuning"),
]

print("\nv8 Fix Verification:")
for pattern, desc in checks:
    found = pattern in full_src
    print(f"  {'PASS' if found else 'FAIL'}: {desc} ({pattern})")

# Check NO bad patterns
bad = [
    ("use_amp = False", "AMP disable logic"),
    ("nan_recovery", "NaN recovery system"),
    ("CATASTROPHIC", "Catastrophic rollback"),
]

print("\nRemoved Bad Patterns:")
for pattern, desc in bad:
    found = pattern in full_src
    print(f"  {'FAIL - STILL PRESENT' if found else 'PASS - REMOVED'}: {desc}")
