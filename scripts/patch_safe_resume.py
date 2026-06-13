import json, sys
sys.stdout.reconfigure(encoding='utf-8')

NB_PATH = r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\livertumor-model.ipynb'
nb = json.load(open(NB_PATH, encoding='utf-8'))
src = ''.join(nb['cells'][12]['source'])

# Add safety: try/except around optimizer.load_state_dict for Adam→AdamW compatibility
old_optimizer_load = """        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # v6: ALSO restore GradScaler state
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            print(f"   ✅ GradScaler state restored")"""

new_optimizer_load = """        # v7.2: Safe optimizer restore — handles Adam→AdamW transition
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print(f"   ✅ Optimizer state restored")
        except (ValueError, RuntimeError) as e:
            print(f"   ⚠️ Optimizer state incompatible (likely Adam→AdamW change): {e}")
            print(f"   → Using fresh AdamW optimizer with pretrained weights")
            weights_only_resume = True  # treat as weights-only
        # v6: ALSO restore GradScaler state
        if not weights_only_resume and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            print(f"   ✅ GradScaler state restored")"""

if old_optimizer_load in src:
    src = src.replace(old_optimizer_load, new_optimizer_load)
    print("✅ Added safe optimizer restore (handles Adam→AdamW transition)")
else:
    print("❌ Could not find optimizer load block")
    sys.exit(1)

# Write back
cell_lines = src.split('\n')
new_source = []
for i, line in enumerate(cell_lines):
    if i < len(cell_lines) - 1:
        new_source.append(line + '\n')
    else:
        new_source.append(line)
nb['cells'][12]['source'] = new_source

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✅ Notebook saved")
