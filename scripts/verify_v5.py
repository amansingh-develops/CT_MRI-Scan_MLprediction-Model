"""
v5 Notebook Verification Script
Checks the .ipynb for structural, logical, and consistency issues
before uploading to Kaggle.
"""
import json
import re
import sys

NB_PATH = r"c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\livertumor-model.ipynb"

def load_nb(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_code_cells(nb):
    return [c for c in nb['cells'] if c['cell_type'] == 'code']

def get_all_code(nb):
    return '\n'.join(c['source'] for c in get_code_cells(nb))

def check(name, condition, detail=""):
    status = "✅" if condition else "❌"
    msg = f"{status} {name}"
    if detail and not condition:
        msg += f" — {detail}"
    print(msg)
    return condition

def main():
    nb = load_nb(NB_PATH)
    all_code = get_all_code(nb)
    code_cells = get_code_cells(nb)
    all_ok = True

    print("=" * 60)
    print("  v5 NOTEBOOK VERIFICATION")
    print("=" * 60)

    # ──────────────────────────────────────────────
    # 1. STRUCTURAL CHECKS
    # ──────────────────────────────────────────────
    print("\n📋 STRUCTURAL CHECKS")
    print("-" * 40)

    all_ok &= check("Valid JSON / nbformat", nb.get('nbformat') == 4)
    all_ok &= check("Kaggle metadata present", 'kaggle' in nb.get('metadata', {}))
    kaggle = nb['metadata'].get('kaggle', {})
    all_ok &= check("GPU enabled", kaggle.get('isGpuEnabled', False))
    all_ok &= check("Accelerator = T4", kaggle.get('accelerator') == 'nvidiaTeslaT4')
    all_ok &= check("Has data source", len(kaggle.get('dataSources', [])) > 0)
    all_ok &= check("Has code cells", len(code_cells) >= 6)

    # ──────────────────────────────────────────────
    # 2. PAPER ALIGNMENT CHECKS
    # ──────────────────────────────────────────────
    print("\n📋 PAPER ALIGNMENT CHECKS")
    print("-" * 40)

    all_ok &= check("Optimizer = Adam", 'optim.Adam(' in all_code and 'AdamW' not in all_code)
    all_ok &= check("LR = 1e-4", "LR           = 1e-4" in all_code or "lr=1e-4" in all_code)
    all_ok &= check("Batch = 32", "BATCH        = 32" in all_code)
    all_ok &= check("Epochs = 100", "EPOCHS       = 100" in all_code)
    all_ok &= check("PATIENCE = 10", "PATIENCE     = 10" in all_code)
    all_ok &= check("LR_PAT = 5", "LR_PAT       = 5" in all_code)
    all_ok &= check("LR_FACTOR = 0.5", "LR_FACTOR    = 0.5" in all_code)
    all_ok &= check("NO pos_weight", "pos_weight" not in all_code.replace("NO pos_weight", "").replace("no pos_weight", "").replace("no_pos_weight", ""))
    all_ok &= check("0.5 * BCE + 0.5 * Dice", "0.5 * self.bce" in all_code and "0.5 * self.dice" in all_code)
    all_ok &= check("BCEWithLogitsLoss (plain)", "nn.BCEWithLogitsLoss()" in all_code)
    all_ok &= check("NO sigmoid in model forward", "return self.head(x)" in all_code)
    all_ok &= check("sigmoid in DiceLoss", "torch.sigmoid(pred)" in all_code)

    # ──────────────────────────────────────────────
    # 3. ARCHITECTURE CHECKS
    # ──────────────────────────────────────────────
    print("\n📋 ARCHITECTURE CHECKS")
    print("-" * 40)

    all_ok &= check("Encoder: 64→128→256→512", "feat=[64, 128, 256, 512]" in all_code)
    all_ok &= check("Bottleneck: 1024", "feat[3] * 2" in all_code)
    all_ok &= check("DoubleConv has bias=False", "bias=False" in all_code)
    all_ok &= check("BatchNorm2d present", "nn.BatchNorm2d" in all_code)
    all_ok &= check("ReLU(inplace=True)", "nn.ReLU(inplace=True)" in all_code)
    all_ok &= check("ConvTranspose2d for upsampling", "nn.ConvTranspose2d" in all_code)
    all_ok &= check("MaxPool2d(2) for downsampling", "nn.MaxPool2d(2)" in all_code)
    all_ok &= check("Kaiming init", "kaiming_normal_" in all_code)
    all_ok &= check("Xavier init for head", "xavier_normal_(self.head.weight)" in all_code)
    all_ok &= check("Output: Conv2d(64, 2, 1)", "nn.Conv2d(feat[0], out_ch, 1)" in all_code)

    # ──────────────────────────────────────────────
    # 4. v5 CRITICAL FIXES
    # ──────────────────────────────────────────────
    print("\n📋 v5 CRITICAL FIXES")
    print("-" * 40)

    all_ok &= check("GradScaler state in LATEST save", all_code.count('"scaler_state_dict": scaler.state_dict()') >= 2,
                     "Must save scaler in both CKPT_LATEST and CKPT_BEST")
    all_ok &= check("GradScaler state restored on resume", '"scaler_state_dict" in ckpt' in all_code)
    all_ok &= check("Fresh scaler on NaN recovery", "scaler = torch.amp.GradScaler('cuda')  # fresh scaler" in all_code)
    all_ok &= check("Fresh optimizer on NaN recovery", "optimizer = optim.Adam(model.parameters(), lr=LR)  # fresh optimizer" in all_code)
    all_ok &= check("AMP disabled during recovery", 'nan_recovery = True' in all_code and 'use_amp = False' in all_code)
    all_ok &= check("Inf gradient detection", "torch.isnan(grad_norm) or torch.isinf(grad_norm)" in all_code)
    all_ok &= check("Warmup LR implemented", "WARMUP_EPOCHS" in all_code and "get_warmup_lr" in all_code)
    all_ok &= check("Gradient norm tracking", "max_grad_norm" in all_code)
    all_ok &= check("Scheduler skipped during warmup", "epoch > WARMUP_EPOCHS" in all_code)

    # ──────────────────────────────────────────────
    # 5. DATA PIPELINE CHECKS
    # ──────────────────────────────────────────────
    print("\n📋 DATA PIPELINE CHECKS")
    print("-" * 40)

    all_ok &= check("INTER_NEAREST for masks", "cv2.INTER_NEAREST" in all_code)
    all_ok &= check("INTER_LINEAR for images", "cv2.INTER_LINEAR" in all_code)
    all_ok &= check("Mask binarization (> 0)", "> 0).float()" in all_code)
    all_ok &= check("Image normalization / 255.0", "/ 255.0" in all_code)
    all_ok &= check("Augmentation: horizontal flip", "np.fliplr" in all_code)
    all_ok &= check("Augmentation: vertical flip", "np.flipud" in all_code)
    all_ok &= check("Augmentation: rotation", "np.rot90" in all_code)
    all_ok &= check("Augmentation: brightness/contrast", "alpha = random.uniform" in all_code)
    all_ok &= check("None image fallback", "np.zeros((256, 256)" in all_code)
    all_ok &= check("pin_memory=True", "pin_memory=True" in all_code)

    # ──────────────────────────────────────────────
    # 6. TRAINING LOOP SAFETY
    # ──────────────────────────────────────────────
    print("\n📋 TRAINING LOOP SAFETY")
    print("-" * 40)

    all_ok &= check("Gradient clipping", "clip_grad_norm_" in all_code)
    all_ok &= check("GRAD_CLIP = 1.0", "GRAD_CLIP    = 1.0" in all_code)
    all_ok &= check("NaN loss detection", "torch.isnan(loss) or torch.isinf(loss)" in all_code)
    all_ok &= check("Smart timeout", "MAX_TOTAL_HOURS" in all_code)
    all_ok &= check("Early stopping", "no_improve >= PATIENCE" in all_code)
    all_ok &= check("ReduceLROnPlateau", "ReduceLROnPlateau" in all_code)
    all_ok &= check("model.train() in training", "model.train()" in all_code)
    all_ok &= check("model.eval() in validation", "model.eval()" in all_code)
    all_ok &= check("torch.no_grad() in validation", "with torch.no_grad():" in all_code)
    all_ok &= check("zero_grad(set_to_none=True)", "set_to_none=True" in all_code)
    all_ok &= check("non_blocking=True on .to(device)", "non_blocking=True" in all_code)

    # ──────────────────────────────────────────────
    # 7. CHECKPOINT & RESUME LOGIC
    # ──────────────────────────────────────────────
    print("\n📋 CHECKPOINT & RESUME LOGIC")
    print("-" * 40)

    all_ok &= check("Auto-scan /kaggle/input/", "/kaggle/input/" in all_code)
    all_ok &= check("weights_only_resume flag", "weights_only_resume" in all_code)
    all_ok &= check("Full resume loads optimizer", 'optimizer.load_state_dict(ckpt["optimizer_state_dict"])' in all_code)
    all_ok &= check("Weights-only resets optimizer", "weights_only_resume = True" in all_code)
    all_ok &= check("Latest checkpoint saved every epoch", "CKPT_LATEST" in all_code)
    all_ok &= check("Best checkpoint saved on improvement", "CKPT_BEST" in all_code)
    all_ok &= check("Training log saved every epoch", 'training_log.csv' in all_code)

    # ──────────────────────────────────────────────
    # 8. OUTPUT CELLS
    # ──────────────────────────────────────────────
    print("\n📋 OUTPUT & VISUALIZATION")
    print("-" * 40)

    all_ok &= check("Training curves cell exists", "training_curves.png" in all_code)
    all_ok &= check("Prediction visualization exists", "predictions.png" in all_code)
    all_ok &= check("Data diagnostic plot exists", "data_diagnostic.png" in all_code)
    all_ok &= check("Overfit test exists", "OVERFIT TEST" in all_code)
    all_ok &= check("8 sanity checks", "ALL 8 CHECKS PASSED" in all_code)

    # ──────────────────────────────────────────────
    # 9. POTENTIAL GOTCHAS
    # ──────────────────────────────────────────────
    print("\n📋 POTENTIAL GOTCHA CHECKS")
    print("-" * 40)

    # Check there's no leftover pos_weight anywhere (allowing in comments)
    pos_weight_lines = [line.strip() for line in all_code.split('\n')
                        if 'pos_weight' in line.lower()
                        and not line.strip().startswith('#')
                        and not line.strip().startswith('"')
                        and not line.strip().startswith("'")]
    # Filter to only actual code usage (not comments in strings)
    real_pos_weight = [l for l in pos_weight_lines if '=' in l and 'pos_weight' in l.split('=')[0]]
    all_ok &= check("No pos_weight assignment in code", len(real_pos_weight) == 0,
                     f"Found: {real_pos_weight}")

    # Check no SGD
    all_ok &= check("No SGD optimizer", "optim.SGD" not in all_code)

    # Check no AdamW
    all_ok &= check("No AdamW optimizer", "optim.AdamW" not in all_code and "AdamW" not in all_code)

    # Check overfit test uses different lr than training
    all_ok &= check("Overfit test uses higher lr (1e-3)", "lr=1e-3)  # higher for overfit test" in all_code or "lr=1e-3)" in all_code)

    # Verify dataset paths are Kaggle paths
    all_ok &= check("Data paths are Kaggle paths", "/kaggle/input/" in all_code)

    # Check for common Python bugs
    all_ok &= check("No 'import *'", "import *" not in all_code)

    # Verify the training cell doesn't re-import (redefine globals)
    train_cell = code_cells[4]['source'] if len(code_cells) > 4 else ""
    all_ok &= check("Training cell uses existing variables", "DATA_ROOT" not in train_cell, 
                     "Training cell should not redefine DATA_ROOT")

    # ──────────────────────────────────────────────
    # 10. LOGIC FLOW VERIFICATION
    # ──────────────────────────────────────────────
    print("\n📋 LOGIC FLOW VERIFICATION")
    print("-" * 40)

    # Verify warmup doesn't apply when resuming from previous version
    all_ok &= check("Warmup skipped on weights-only resume",
                     "not weights_only_resume" in all_code,
                     "Warmup should be skipped when resuming pretrained weights")

    # Verify scheduler doesn't step during warmup
    all_ok &= check("Scheduler only after warmup",
                     "if epoch > WARMUP_EPOCHS:" in all_code)

    # Verify nan_recovery flag properly cycles
    all_ok &= check("nan_recovery resets on success",
                     "nan_recovery = False  # successful epoch" in all_code)

    # Verify the overfit test cleans up GPU memory
    all_ok &= check("Overfit test cleans up memory",
                     "del test_model, test_criterion, test_opt" in all_code)

    # Verify gradient norm is a tensor before calling .item()
    all_ok &= check("Grad norm .item() guarded",
                     "not (torch.isnan(grad_norm) or torch.isinf(grad_norm))" in all_code)

    # ──────────────────────────────────────────────
    # FINAL SUMMARY
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_ok:
        print("  ✅ ALL CHECKS PASSED — GOOD TO GO!")
    else:
        print("  ❌ SOME CHECKS FAILED — Review issues above")
    print("=" * 60)

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
