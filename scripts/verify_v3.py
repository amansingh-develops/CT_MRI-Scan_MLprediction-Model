import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import json

with open('livertumor-model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Quick checks
src = nb['cells'][12]['source']

checks = {
    "Smart timeout": "avg_epoch_time_hrs" in src,
    "Latest checkpoint save": "CKPT_LATEST" in src,
    "Best checkpoint save": "CKPT_BEST" in src,
    "Auto-resume from latest": "resume_ckpt" in src,
    "Remaining time calc": "remaining_hrs" in src,
    "Mask fix still present": "> 0).float()" in nb['cells'][6]['source'],
    "Old mask bug absent": "/ 255.0).unsqueeze(0) > 0.5" not in nb['cells'][6]['source'],
    "pos_weight in training": "POS_WEIGHT_LIVER" in src,
    "AMP enabled": "autocast" in src,
    "Gradient clipping": "clip_grad_norm_" in src,
    "Training log CSV": "training_log.csv" in src,
}

all_pass = True
for name, result in checks.items():
    status = "PASS" if result else "FAIL"
    if not result:
        all_pass = False
    print(f"  [{status}] {name}")

print(f"\nAll checks: {'PASSED' if all_pass else 'SOME FAILED!'}")

# Count critical features
print(f"\nNotebook cells: {len(nb['cells'])}")
print(f"Training cell lines: {src.count(chr(10)) + 1}")
