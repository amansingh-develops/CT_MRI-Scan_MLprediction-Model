import json, sys
sys.stdout.reconfigure(encoding='utf-8')

NB_PATH = r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\livertumor-model.ipynb'

nb = json.load(open(NB_PATH, encoding='utf-8'))
cell = ''.join(nb['cells'][12]['source'])

# 1. Remove MAX_TOTAL_HOURS config line
cell = cell.replace("MAX_TOTAL_HOURS = 11.5       # Kaggle GPU time limit safety\n", "")

# 2. Remove the timeout print line
cell = cell.replace('print(f"Timeout:      SMART — stops when next epoch won\'t fit in {MAX_TOTAL_HOURS}h")\n', "")

# 3. Remove the entire SMART TIMEOUT block
timeout_block = '''    # ★ SMART TIMEOUT
    total_elapsed_hrs = (time.time() - global_t0) / 3600.0
    avg_epoch_hrs = np.mean(epoch_times) / 3600.0
    remaining_hrs = MAX_TOTAL_HOURS - total_elapsed_hrs

    if remaining_hrs < avg_epoch_hrs * 1.5:
        print(f"\\n⏰ SMART TIMEOUT: {total_elapsed_hrs:.2f}h elapsed, "
              f"avg epoch={avg_epoch_hrs*60:.1f}min, "
              f"remaining={remaining_hrs*60:.1f}min")
        print(f"   Stopping gracefully. Resume by re-running.")
        break'''

cell = cell.replace(timeout_block, "")

# Write back
new_lines = [line + '\n' for line in cell.split('\n')]
if new_lines and new_lines[-1].strip() == '':
    new_lines[-1] = '\n'
nb['cells'][12]['source'] = new_lines

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Verify
final = ''.join(new_lines)
assert 'MAX_TOTAL_HOURS' not in final, "MAX_TOTAL_HOURS still present!"
assert 'SMART TIMEOUT' not in final, "SMART TIMEOUT still present!"
assert 'no_improve >= PATIENCE' in final, "Early stopping missing!"

print("✅ Timeout removed")
print("✅ Early stopping still active (patience=10)")
print("✅ Model will only stop when it stops improving")
