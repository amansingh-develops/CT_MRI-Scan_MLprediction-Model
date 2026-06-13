#!/usr/bin/env python3
"""
v7 "Final" Patch — Apply all 7 fixes to Cell 12 of livertumor-model.ipynb

Changes:
  1. GradScaler init_scale=1024 (was default 65536)
  2. Remove scaler reset on Inf cascade (let it adapt)
  3. Adaptive gradient clipping (AMP=5.0, FP32=50.0)
  4. Adam → AdamW with weight_decay=1e-5
  5. init_scale=1024 in catastrophic rollback
  6. init_scale=1024 in NaN recovery
  7. Match validation precision to training precision
"""
import json, sys, os, shutil

NB_PATH = r'c:\Users\Aman Singh\OneDrive - Noida Institute of Engineering and Technology\Desktop\liver_ai_project\livertumor-model.ipynb'

def apply_v7():
    # Backup
    backup = NB_PATH + '.v6_backup'
    if not os.path.exists(backup):
        shutil.copy2(NB_PATH, backup)
        print(f"✅ Backup saved: {backup}")
    
    nb = json.load(open(NB_PATH, encoding='utf-8'))
    src = ''.join(nb['cells'][12]['source'])
    original = src  # keep for comparison
    
    changes = 0
    
    # ─────────────────────────────────────────────
    # CHANGE 1 & 3: Config block — replace GRAD_CLIP, update version header
    # ─────────────────────────────────────────────
    old_config = """# ═══════════════════════════════════════════════════════════════
# CONFIG — v6: Ironclad Training Pipeline
# ═══════════════════════════════════════════════════════════════
BATCH        = 32            # PAPER: batch 32 > 16, 64, 128
EPOCHS       = 100           # PAPER: model stabilizes ~epoch 60
LR           = 1e-4          # PAPER: standard for Adam on medical segmentation
GRAD_CLIP    = 5.0           # FIX #2: healthy norms are 7-12, clip=1.0 was crushing ALL gradients
PATIENCE     = 10            # PAPER: early stopping patience
LR_PAT       = 5             # PAPER: LR scheduler patience
LR_FACTOR    = 0.5           # PAPER: halve LR on plateau
WORKERS      = 2
WARMUP_EPOCHS = 3            # v6: warmup 3 epochs from any start point
MAX_INF_PER_EPOCH = 5        # v6: if >5 Inf batches, switch to FP32 mid-epoch"""
    
    new_config = """# ═══════════════════════════════════════════════════════════════
# CONFIG — v7: Final Training Pipeline
# ═══════════════════════════════════════════════════════════════
BATCH          = 32          # PAPER: batch 32 > 16, 64, 128
EPOCHS         = 100         # PAPER: model stabilizes ~epoch 60
LR             = 1e-4        # PAPER: standard for Adam on medical segmentation
GRAD_CLIP_AMP  = 5.0         # v7: AMP gradient norms are 2-12
GRAD_CLIP_FP32 = 50.0        # v7: FP32 gradient norms are 17-100
PATIENCE       = 10          # PAPER: early stopping patience
LR_PAT         = 5           # PAPER: LR scheduler patience
LR_FACTOR      = 0.5         # PAPER: halve LR on plateau
WORKERS        = 2
WARMUP_EPOCHS  = 3           # v6: warmup 3 epochs from any start point
MAX_INF_PER_EPOCH = 5        # v6: if >5 Inf batches, switch to FP32 mid-epoch"""
    
    if old_config in src:
        src = src.replace(old_config, new_config)
        changes += 1
        print("✅ Change 1+3a: Config updated (version header, adaptive GRAD_CLIP)")
    else:
        print("❌ FAILED: Could not find config block")
        return False
    
    # ─────────────────────────────────────────────
    # CHANGE 3b: Update grad clip references in training loop (AMP path)
    # ─────────────────────────────────────────────
    old_amp_clip = "            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)\n"
    new_amp_clip = "            clip_val = GRAD_CLIP_AMP if use_amp else GRAD_CLIP_FP32\n            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)\n"
    
    # There are two places: one in AMP path (line 198) and one in FP32 path (line 223)
    # The AMP one is after scaler.unscale_(optimizer)
    amp_context = "            scaler.unscale_(optimizer)\n            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)\n"
    amp_replacement = "            scaler.unscale_(optimizer)\n            clip_val = GRAD_CLIP_AMP if use_amp else GRAD_CLIP_FP32\n            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)\n"
    
    if amp_context in src:
        src = src.replace(amp_context, amp_replacement)
        changes += 1
        print("✅ Change 3b: AMP gradient clip → adaptive")
    else:
        print("❌ FAILED: Could not find AMP grad clip")
        return False
    
    # FP32 path clip
    fp32_clip_old = "            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)\n            optimizer.step()\n"
    fp32_clip_new = "            clip_val = GRAD_CLIP_AMP if use_amp else GRAD_CLIP_FP32\n            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)\n            optimizer.step()\n"
    
    if fp32_clip_old in src:
        src = src.replace(fp32_clip_old, fp32_clip_new)
        changes += 1
        print("✅ Change 3c: FP32 gradient clip → adaptive")
    else:
        print("❌ FAILED: Could not find FP32 grad clip")
        return False
    
    # ─────────────────────────────────────────────
    # CHANGE 1b + 4: GradScaler init + Adam → AdamW
    # ─────────────────────────────────────────────
    old_init = """optimizer = optim.Adam(model.parameters(), lr=LR)  # plain Adam, paper-proven
scaler    = torch.amp.GradScaler('cuda')"""
    new_init = """optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)  # v7: AdamW for regularization
scaler    = torch.amp.GradScaler('cuda', init_scale=1024)  # v7: lower init_scale prevents Inf spiral"""
    
    if old_init in src:
        src = src.replace(old_init, new_init)
        changes += 1
        print("✅ Change 1b+4: Adam → AdamW, GradScaler init_scale=1024")
    else:
        print("❌ FAILED: Could not find optimizer/scaler init")
        return False
    
    # ─────────────────────────────────────────────
    # CHANGE 2: Remove scaler reset on Inf cascade
    # ─────────────────────────────────────────────
    old_cascade = """                if inf_count_this_epoch >= MAX_INF_PER_EPOCH:
                    print(f"\\n  🛡️ Inf cascade detected ({inf_count_this_epoch} events) — disabling AMP for rest of epoch")
                    use_amp = False
                    # Reset scaler for next epoch
                    scaler = torch.amp.GradScaler('cuda')"""
    new_cascade = """                if inf_count_this_epoch >= MAX_INF_PER_EPOCH:
                    print(f"\\n  🛡️ Inf cascade detected ({inf_count_this_epoch} events) — disabling AMP for rest of epoch")
                    use_amp = False
                    # v7: DON'T reset scaler — let it adapt its scale down naturally
                    # The scaler.update() calls above already halve the scale on each Inf"""
    
    if old_cascade in src:
        src = src.replace(old_cascade, new_cascade)
        changes += 1
        print("✅ Change 2: Removed scaler reset on cascade (let it adapt)")
    else:
        print("❌ FAILED: Could not find cascade handler")
        return False
    
    # ─────────────────────────────────────────────
    # CHANGE 6: NaN recovery — init_scale=1024 + AdamW
    # ─────────────────────────────────────────────
    old_nan_recovery = """        # v6: FULL RECOVERY — reset scaler, load best model
        scaler = torch.amp.GradScaler('cuda')  # fresh scaler
        if os.path.exists(CKPT_BEST):
            ckpt = torch.load(CKPT_BEST, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer = optim.Adam(model.parameters(), lr=LR * 0.5)  # lower LR"""
    
    new_nan_recovery = """        # v7: FULL RECOVERY — conservative scaler, load best model
        scaler = torch.amp.GradScaler('cuda', init_scale=1024)  # v7: lower init_scale
        if os.path.exists(CKPT_BEST):
            ckpt = torch.load(CKPT_BEST, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer = optim.AdamW(model.parameters(), lr=LR * 0.5, weight_decay=1e-5)  # v7: AdamW"""
    
    if old_nan_recovery in src:
        src = src.replace(old_nan_recovery, new_nan_recovery)
        changes += 1
        print("✅ Change 6: NaN recovery uses init_scale=1024 + AdamW")
    else:
        print("❌ FAILED: Could not find NaN recovery block")
        return False
    
    # ─────────────────────────────────────────────
    # CHANGE 5: Catastrophic rollback — init_scale=1024 + AdamW
    # ─────────────────────────────────────────────
    old_cat_rollback = """            # Use a lower LR after catastrophe to prevent recurrence
            recovery_lr = LR * 0.5
            optimizer = optim.Adam(model.parameters(), lr=recovery_lr)
            scaler = torch.amp.GradScaler('cuda')"""
    
    new_cat_rollback = """            # Use a lower LR after catastrophe to prevent recurrence
            recovery_lr = LR * 0.5
            optimizer = optim.AdamW(model.parameters(), lr=recovery_lr, weight_decay=1e-5)  # v7: AdamW
            scaler = torch.amp.GradScaler('cuda', init_scale=1024)  # v7: lower init_scale"""
    
    if old_cat_rollback in src:
        src = src.replace(old_cat_rollback, new_cat_rollback)
        changes += 1
        print("✅ Change 5: Catastrophic rollback uses init_scale=1024 + AdamW")
    else:
        print("❌ FAILED: Could not find catastrophic rollback block")
        return False
    
    # ─────────────────────────────────────────────
    # CHANGE 7: Match validation precision to training
    # ─────────────────────────────────────────────
    old_val = """    with torch.no_grad():
        for imgs, msks in tqdm(val_ld, desc=f"E{epoch:02d} Val", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                loss  = criterion(preds, msks)"""
    
    new_val = """    with torch.no_grad():
        for imgs, msks in tqdm(val_ld, desc=f"E{epoch:02d} Val", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)
            # v7: match val precision to training (AMP or FP32)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    preds = model(imgs)
                    loss  = criterion(preds, msks)
            else:
                preds = model(imgs)
                loss  = criterion(preds, msks)"""
    
    if old_val in src:
        src = src.replace(old_val, new_val)
        changes += 1
        print("✅ Change 7: Validation precision matches training mode")
    else:
        print("❌ FAILED: Could not find validation loop")
        return False
    
    # ─────────────────────────────────────────────
    # UPDATE: Print header
    # ─────────────────────────────────────────────
    src = src.replace(
        '  LIVER AI — v6 Ironclad Training Pipeline',
        '  LIVER AI — v7 Final Training Pipeline'
    )
    src = src.replace(
        'print(f"Optimizer:    Adam (lr={LR}) — paper-proven")',
        'print(f"Optimizer:    AdamW (lr={LR}, wd=1e-5) — v7")'
    )
    src = src.replace(
        'print(f"Grad clip:    {GRAD_CLIP} (healthy norms are 7-12)")',
        'print(f"Grad clip:    AMP={GRAD_CLIP_AMP} / FP32={GRAD_CLIP_FP32} (adaptive)")'
    )
    changes += 1
    print("✅ Header updated to v7")
    
    # ─────────────────────────────────────────────
    # WRITE BACK
    # ─────────────────────────────────────────────
    # Convert back to cell format
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
    
    print(f"\n{'='*60}")
    print(f"  ✅ ALL {changes} CHANGES APPLIED SUCCESSFULLY")
    print(f"  Backup at: {backup}")
    print(f"{'='*60}")
    return True

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    success = apply_v7()
    if not success:
        print("\n❌ PATCH FAILED — notebook unchanged")
        sys.exit(1)
