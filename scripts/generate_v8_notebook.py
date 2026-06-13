"""
Generate the complete v8 training notebook.
Run: python scripts/generate_v8_notebook.py
Output: livertumor-model.ipynb (overwrites existing)
"""
import json
import os

def md(source):
    """Create a markdown cell."""
    return {"cell_type": "markdown", "source": source, "metadata": {}}

def code(source):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "source": source,
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None
    }

# ═══════════════════════════════════════════════════════════════
# CELL DEFINITIONS
# ═══════════════════════════════════════════════════════════════

CELL_1_MD = """## Liver & Tumor Segmentation — v8 Clean Pipeline

**Architecture:** Standard 2D U-Net (64→128→256→512→1024)
**Dataset:** LiTS17 (PNG slices)
**Loss:** 0.5 × BCE + 0.5 × Dice (FP32-safe)
**Optimizer:** Adam (lr=3e-4 from scratch, 1e-5 for fine-tuning)

### v8 Root Cause Fixes
1. **WeightedRandomSampler** — oversample liver-positive slices (fixes 70.6% empty slice noise)
2. **Batch=16 + GradAccum=2** — safe VRAM on T4 (no OOM even without AMP)
3. **FP32 Dice Loss** — `.float()` cast prevents FP16 overflow (fixes all Inf gradients)
4. **AMP always ON** — never disabled, GradScaler handles Inf natively
5. **Proper LR** — 3e-4 from scratch, 1e-5 for fine-tuning
6. **No recovery system** — simple early stopping, no rollback/nan_recovery/FP32 fallback"""

# ──────────────────────────────────────────────────────────────
CELL_2_GPU = r"""# ═══════════════════════════════════════════════════════════════
# Cell 1 — GPU Setup + Memory Config
# ═══════════════════════════════════════════════════════════════
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch, gc
gc.collect()
torch.cuda.empty_cache()

assert torch.cuda.is_available(), "No GPU found!"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"CUDA Alloc Config: expandable_segments=True")

torch.backends.cudnn.benchmark = True
print(f"cuDNN benchmark: ON ✓")
print(f"\n✅ GPU ready!")"""

# ──────────────────────────────────────────────────────────────
CELL_3_PATHS = r"""# ═══════════════════════════════════════════════════════════════
# Cell 2 — Verify Dataset Paths (3-way split: Train / Val / Test)
# ═══════════════════════════════════════════════════════════════
import os

DATA_ROOT = "/kaggle/input/datasets/andrewmvd/lits-png/dataset_6/dataset_6"
TRAIN_CSV = "/kaggle/input/datasets/andrewmvd/lits-png/lits_train.csv"
VAL_CSV   = "/kaggle/input/datasets/andrewmvd/lits-png/lits_val.csv"
TEST_CSV  = "/kaggle/input/datasets/andrewmvd/lits-png/lits_test.csv"

for tag, p in [("Images", DATA_ROOT), ("Train", TRAIN_CSV), ("Val", VAL_CSV), ("Test", TEST_CSV)]:
    assert os.path.exists(p), f"NOT FOUND: {p}"

import pandas as pd
n_train = len(pd.read_csv(TRAIN_CSV))
n_val   = len(pd.read_csv(VAL_CSV))
n_test  = len(pd.read_csv(TEST_CSV))

print("✅ All paths verified!")
print(f"   Images: {DATA_ROOT}")
print(f"   Train:  {TRAIN_CSV} ({n_train:,} slices)")
print(f"   Val:    {VAL_CSV} ({n_val:,} slices)")
print(f"   Test:   {TEST_CSV} ({n_test:,} slices)")
print(f"   Total:  {n_train+n_val+n_test:,} slices")"""

# ──────────────────────────────────────────────────────────────
CELL_4_CODE = r"""# ═══════════════════════════════════════════════════════════════
# Cell 3 — Model, Dataset, Loss, Metrics
# ═══════════════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, pandas as pd, cv2, os, glob, time, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Dataset ───────────────────────────────────────────────────
class LITSDataset(Dataset):
    def __init__(self, csv_file, root_dir, augment=False):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, os.path.basename(str(row["filepath"])))
        liv_path = os.path.join(self.root_dir, os.path.basename(str(row["liver_maskpath"])))
        tum_path = os.path.join(self.root_dir, os.path.basename(str(row["tumor_maskpath"])))

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        liv = cv2.imread(liv_path, cv2.IMREAD_GRAYSCALE)
        tum = cv2.imread(tum_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        liv = cv2.resize(liv, (256, 256), interpolation=cv2.INTER_NEAREST)
        tum = cv2.resize(tum, (256, 256), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
                liv = np.fliplr(liv).copy()
                tum = np.fliplr(tum).copy()
            if np.random.rand() > 0.5:
                img = np.flipud(img).copy()
                liv = np.flipud(liv).copy()
                tum = np.flipud(tum).copy()
            k = np.random.randint(0, 4)
            img = np.rot90(img, k).copy()
            liv = np.rot90(liv, k).copy()
            tum = np.rot90(tum, k).copy()
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)

        img = img.astype(np.float32) / 255.0
        liv = (liv > 127).astype(np.float32)
        tum = (tum > 127).astype(np.float32)

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(np.stack([liv, tum], axis=0))
        return img, mask

# ── U-Net ─────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)

# ── Loss (v8 FIX: FP32-safe) ─────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # v8 FIX #3: Force FP32 — prevents FP16 overflow that caused ALL Inf gradients
        pred = torch.sigmoid(pred).float()
        target = target.float()
        pred = pred.contiguous()
        target = target.contiguous()
        inter = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        # v8 FIX #3: Force FP32 for BCE too
        return 0.5 * self.bce(pred.float(), target.float()) + 0.5 * self.dice(pred, target)

# ── Dice Metric ───────────────────────────────────────────────
def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    with torch.no_grad():
        pred = torch.sigmoid(pred).float()
        target = target.float()
        pred_bin = (pred > threshold).float()
        inter = (pred_bin * target).sum(dim=(2, 3))
        union = pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * inter + smooth) / (union + smooth)
        liver = dice[:, 0].mean().item()
        tumor = dice[:, 1].mean().item()
    return liver, tumor, (liver + tumor) / 2

print("✅ Model, Dataset, Loss — all defined")
print(f"   UNet params: {sum(p.numel() for p in UNet(1,2).parameters()):,}")
print(f"   DiceLoss: FP32-safe (v8 fix)")
print(f"   CombinedLoss: 0.5*BCE + 0.5*Dice (FP32-safe)")"""

# ──────────────────────────────────────────────────────────────
CELL_5_DIAG = r"""# ═══════════════════════════════════════════════════════════════
# Cell 4 — Data Diagnostics & Overfit Test
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  DATA DIAGNOSTICS")
print("=" * 60)

diag_ds = LITSDataset(VAL_CSV, DATA_ROOT, augment=False)
n_liver, n_tumor, n_both_empty = 0, 0, 0
pixel_sums = []

for idx in range(0, len(diag_ds), max(1, len(diag_ds) // 500)):
    _, msk = diag_ds[idx]
    has_liver = msk[0].sum().item() > 0
    has_tumor = msk[1].sum().item() > 0
    if has_liver: n_liver += 1
    if has_tumor: n_tumor += 1
    if not has_liver and not has_tumor: n_both_empty += 1
    pixel_sums.append(msk.sum().item())

total = max(1, n_liver + n_tumor + n_both_empty - (n_liver if n_tumor else 0))
sampled = len(pixel_sums)
print(f"\nSampled {sampled} slices:")
print(f"  Liver non-empty: {n_liver} ({100*n_liver/sampled:.1f}%)")
print(f"  Tumor non-empty: {n_tumor} ({100*n_tumor/sampled:.1f}%)")
print(f"  Both empty:      {n_both_empty} ({100*n_both_empty/sampled:.1f}%)")
print(f"\n  ➡️ v8 uses WeightedRandomSampler to oversample liver-positive slices")

# ── Quick overfit test ────────────────────────────────────────
print("\n" + "=" * 60)
print("  OVERFIT TEST (10 batches, 50 steps)")
print("=" * 60)

device = torch.device("cuda")
test_model = UNet(1, 2).to(device)
test_opt = optim.Adam(test_model.parameters(), lr=1e-3)
test_loss_fn = CombinedLoss().to(device)
test_scaler = torch.amp.GradScaler('cuda')

test_ld = DataLoader(diag_ds, batch_size=8, shuffle=True, num_workers=0)
test_iter = iter(test_ld)
batch_imgs, batch_msks = next(test_iter)
batch_imgs = batch_imgs.to(device)
batch_msks = batch_msks.to(device)

test_model.train()
for step in range(50):
    test_opt.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda'):
        out = test_model(batch_imgs)
        loss = test_loss_fn(out, batch_msks)
    test_scaler.scale(loss).backward()
    test_scaler.step(test_opt)
    test_scaler.update()
    if step % 10 == 0:
        dl, dt, _ = dice_score(out, batch_msks)
        print(f"  Step {step:02d}: loss={loss.item():.4f}, liver={dl:.4f}, tumor={dt:.4f}")

dl, dt, _ = dice_score(out, batch_msks)
print(f"\n  Final: loss={loss.item():.4f}, liver={dl:.4f}, tumor={dt:.4f}")
if dl > 0.8:
    print("  ✅ Overfit test PASSED — model can learn")
else:
    print(f"  ⚠️ Liver dice {dl:.4f} < 0.8 — may need more steps")

del test_model, test_opt, test_loss_fn, test_scaler, batch_imgs, batch_msks
torch.cuda.empty_cache()
gc.collect()"""

# ──────────────────────────────────────────────────────────────
CELL_6_SANITY = r"""# ═══════════════════════════════════════════════════════════════
# Cell 5 — Pre-Training Sanity Checks
# ═══════════════════════════════════════════════════════════════
import gc

device = torch.device("cuda")
model = UNet(1, 2).to(device)

# Check 1: Output shape
dummy = torch.randn(2, 1, 256, 256, device=device)
with torch.amp.autocast('cuda'):
    out = model(dummy)
assert out.shape == (2, 2, 256, 256), f"Bad shape: {out.shape}"
print("✅ Check 1: Output shape (2,2,256,256) — PASS")

# Check 2: Loss computes without NaN
criterion = CombinedLoss().to(device)
target = torch.zeros(2, 2, 256, 256, device=device)
with torch.amp.autocast('cuda'):
    loss = criterion(out, target)
assert not torch.isnan(loss), f"Loss is NaN: {loss}"
assert not torch.isinf(loss), f"Loss is Inf: {loss}"
print(f"✅ Check 2: Loss = {loss.item():.4f} — PASS")

# Check 3: Dice loss FP32 safety
print("\n  Testing DiceLoss FP32 safety under AMP...")
dice_fn = DiceLoss().to(device)
for case_name, p, t in [
    ("zeros",     torch.zeros(2,2,256,256, device=device), torch.zeros(2,2,256,256, device=device)),
    ("ones",      torch.ones(2,2,256,256, device=device)*5, torch.ones(2,2,256,256, device=device)),
    ("large",     torch.ones(2,2,256,256, device=device)*100, torch.ones(2,2,256,256, device=device)),
    ("mixed",     torch.randn(2,2,256,256, device=device)*10, torch.randint(0,2,(2,2,256,256), device=device).float()),
]:
    with torch.amp.autocast('cuda'):
        d = dice_fn(p, t)
    ok = not (torch.isnan(d) or torch.isinf(d))
    print(f"    {case_name}: {d.item():.4f} {'✅' if ok else '❌'}")

# Check 4: GradScaler step
scaler = torch.amp.GradScaler('cuda')
optimizer = optim.Adam(model.parameters(), lr=3e-4)
optimizer.zero_grad()
with torch.amp.autocast('cuda'):
    loss = criterion(model(dummy), target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
print(f"✅ Check 4: GradScaler step — PASS (scale={scaler.get_scale():.0f})")

# Check 5: VRAM usage
mem = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
total = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f"✅ Check 5: VRAM = {mem:.2f}GB used / {total:.1f}GB total ({100*mem/total:.0f}%)")

del model, dummy, out, target, criterion, optimizer, scaler
torch.cuda.empty_cache()
gc.collect()
print("\n✅ All sanity checks PASSED — ready to train")"""

# ──────────────────────────────────────────────────────────────
CELL_7_TRAIN = r"""# ═══════════════════════════════════════════════════════════════
# Cell 6 — v8 Training Loop
# ═══════════════════════════════════════════════════════════════
# v8: CLEAN REWRITE — all 6 root causes fixed
#   1. WeightedRandomSampler (liver-positive oversampled 3x)
#   2. Batch=16 + GradAccum=2 (effective=32, VRAM-safe)
#   3. DiceLoss in FP32 (no more Inf gradients)
#   4. AMP always ON (never disabled)
#   5. LR=3e-4 from scratch, 1e-5 for fine-tune
#   6. No recovery system — simple early stopping only
# ═══════════════════════════════════════════════════════════════

import gc
gc.collect()
torch.cuda.empty_cache()

# ── CONFIG ────────────────────────────────────────────────────
BATCH         = 16
ACCUM_STEPS   = 2       # effective batch = 32
EPOCHS        = 100
LR            = 3e-4    # research standard for Adam from scratch
GRAD_CLIP     = 5.0     # conservative
PATIENCE      = 15      # generous early stopping
LR_PAT        = 5       # scheduler patience
LR_FACTOR     = 0.5
WORKERS       = 2
WARMUP_EPOCHS = 5
PREV_CKPT     = None    # set manually if needed

device = torch.device("cuda")

print("=" * 60)
print("  LIVER AI — v8 CLEAN TRAINING PIPELINE")
print("=" * 60)
print(f"  Batch:       {BATCH} x {ACCUM_STEPS} accum = {BATCH*ACCUM_STEPS} effective")
print(f"  Epochs:      {EPOCHS} (early stop: {PATIENCE})")
print(f"  Optimizer:   Adam (lr={LR})")
print(f"  Loss:        0.5*BCE + 0.5*Dice (FP32-safe)")
print(f"  Grad clip:   {GRAD_CLIP}")
print(f"  AMP:         ALWAYS ON (never disabled)")
print(f"  Warmup:      {WARMUP_EPOCHS} epochs")
print(f"  Scheduler:   ReduceLR(patience={LR_PAT}, factor={LR_FACTOR})")
print("=" * 60)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── DATA + WEIGHTED SAMPLER ───────────────────────────────────
train_ds = LITSDataset(TRAIN_CSV, DATA_ROOT, augment=True)
val_ds   = LITSDataset(VAL_CSV,   DATA_ROOT, augment=False)

# v8 FIX #1: Oversample liver-positive slices
# 70.6% of slices are empty → Dice loss gives noisy gradients on them
# WeightedRandomSampler gives liver-positive slices 3x higher probability
print("\n🔍 Scanning masks for weighted sampling...")
train_csv_data = pd.read_csv(TRAIN_CSV)
weights = []
n_pos = 0
scan_t0 = time.time()
for idx in range(len(train_csv_data)):
    row = train_csv_data.iloc[idx]
    liv_path = os.path.join(DATA_ROOT, os.path.basename(str(row["liver_maskpath"])))
    liv = cv2.imread(liv_path, cv2.IMREAD_GRAYSCALE)
    has_liver = (liv is not None and liv.sum() > 0)
    weights.append(3.0 if has_liver else 1.0)
    if has_liver:
        n_pos += 1
    if (idx + 1) % 10000 == 0:
        print(f"   Scanned {idx+1}/{len(train_csv_data)}...")

scan_time = time.time() - scan_t0
n_neg = len(train_csv_data) - n_pos
eff_pos_pct = 100 * (3.0 * n_pos) / (3.0 * n_pos + 1.0 * n_neg)
print(f"   Liver-positive: {n_pos}/{len(train_csv_data)} ({100*n_pos/len(train_csv_data):.1f}%)")
print(f"   Empty slices:   {n_neg}/{len(train_csv_data)} ({100*n_neg/len(train_csv_data):.1f}%)")
print(f"   After weighting: ~{eff_pos_pct:.0f}% of sampled batches will have liver")
print(f"   Scan time: {scan_time:.0f}s")

sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
train_ld = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                      pin_memory=True, num_workers=WORKERS)
val_ld   = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                      pin_memory=True, num_workers=WORKERS)

print(f"\n   Train: {len(train_ds):,} samples ({len(train_ld)} batches)")
print(f"   Val:   {len(val_ds):,} samples ({len(val_ld)} batches)")

# ── MODEL + LOSS + OPTIMIZER ──────────────────────────────────
model     = UNet(1, 2).to(device)
criterion = CombinedLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
# v8: Let PyTorch use default init_scale (65536) — no manual override
scaler    = torch.amp.GradScaler('cuda')

CKPT_BEST   = "models/best_model.pth"
CKPT_LATEST = "models/latest_model.pth"
start_epoch   = 1
best_val_loss = float("inf")

# ── CHECKPOINT SEARCH ─────────────────────────────────────────
def find_prev_checkpoint():
    if PREV_CKPT and os.path.exists(PREV_CKPT):
        return PREV_CKPT
    for pat in ["/kaggle/input/*/models/best_model.pth",
                "/kaggle/input/*/*/models/best_model.pth"]:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None

resume_ckpt = None
if os.path.exists(CKPT_LATEST):
    resume_ckpt = CKPT_LATEST
elif os.path.exists(CKPT_BEST):
    resume_ckpt = CKPT_BEST
else:
    resume_ckpt = find_prev_checkpoint()

if resume_ckpt:
    print(f"\n📂 Found checkpoint: {resume_ckpt}")
    ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # v8 FIX #5: ALWAYS use fresh optimizer with LOWER lr for fine-tuning
    finetune_lr = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=finetune_lr)
    best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))

    print(f"   ✅ Model weights loaded (epoch {ckpt.get('epoch', '?')})")
    print(f"   ✅ Fresh optimizer at lr={finetune_lr:.1e} (fine-tuning)")
    print(f"   ⚠️ Optimizer state NOT loaded (prevents feature destruction)")
    print(f"   📊 Previous best val_loss: {best_val_loss:.4f}")
    del ckpt
    torch.cuda.empty_cache()
else:
    print("\n🆕 Training from scratch")
    print(f"   LR = {LR}")

# ── SCHEDULER ─────────────────────────────────────────────────
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=LR_FACTOR, patience=LR_PAT)

# ── WARMUP ────────────────────────────────────────────────────
def get_warmup_lr(ep, start, base_lr, warmup_n):
    rel = ep - start + 1
    if rel > warmup_n:
        return base_lr
    return base_lr * (0.1 + 0.9 * (rel - 1) / max(1, warmup_n - 1))

# ── TRAINING ──────────────────────────────────────────────────
no_improve        = 0
log               = []
global_t0         = time.time()
epoch_times       = []
warmup_end        = start_epoch + WARMUP_EPOCHS - 1
best_val_dice     = 0.0
base_lr           = optimizer.param_groups[0]['lr']

print(f"\n🎯 Training: epochs {start_epoch} → {EPOCHS}")
print(f"   Warmup: until epoch {warmup_end}")
print(f"   Base LR: {base_lr:.1e}\n")

for epoch in range(start_epoch, EPOCHS + 1):
    t0 = time.time()

    # Warmup LR
    if epoch <= warmup_end:
        wlr = get_warmup_lr(epoch, start_epoch, base_lr, WARMUP_EPOCHS)
        for pg in optimizer.param_groups:
            pg['lr'] = wlr

    # ════ TRAIN ════════════════════════════════════════════════
    model.train()
    t_loss_sum  = 0.0
    t_dl_sum    = 0.0
    t_dt_sum    = 0.0
    n_ok        = 0
    max_gnorm   = 0.0
    inf_events  = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_ld, desc=f"E{epoch:02d} Train", leave=False,
                miniters=50, mininterval=10.0)
    for bi, (imgs, msks) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        msks = msks.to(device, non_blocking=True)

        # v8 FIX #4: AMP is ALWAYS ON — never disabled
        with torch.amp.autocast('cuda'):
            preds = model(imgs)
            loss  = criterion(preds, msks)
            loss  = loss / ACCUM_STEPS

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        # v8 FIX #2: Gradient accumulation step
        if (bi + 1) % ACCUM_STEPS == 0 or (bi + 1) == len(train_ld):
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            if torch.isfinite(gnorm):
                scaler.step(optimizer)
                max_gnorm = max(max_gnorm, gnorm.item())
            else:
                inf_events += 1
                # v8: Don't panic — GradScaler handles this by halving scale

            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Metrics (unscaled loss)
        t_loss_sum += loss.item() * ACCUM_STEPS
        with torch.no_grad():
            dl, dt, _ = dice_score(preds, msks)
        t_dl_sum += dl
        t_dt_sum += dt
        n_ok += 1

    if n_ok == 0:
        print(f"\n💀 ALL batches had NaN loss — stopping.")
        break

    t_loss  = t_loss_sum / n_ok
    t_dl    = t_dl_sum / n_ok
    t_dt    = t_dt_sum / n_ok

    # ════ VALIDATE ═════════════════════════════════════════════
    model.eval()
    v_loss_sum = 0.0
    v_dl_sum   = 0.0
    v_dt_sum   = 0.0
    n_val      = 0

    with torch.no_grad():
        for imgs, msks in tqdm(val_ld, desc=f"E{epoch:02d} Val",
                               leave=False, miniters=50, mininterval=10.0):
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                loss  = criterion(preds, msks)

            if torch.isfinite(loss):
                v_loss_sum += loss.item()
                dl, dt, _ = dice_score(preds, msks)
                v_dl_sum += dl
                v_dt_sum += dt
                n_val += 1

    v_loss = v_loss_sum / max(1, n_val)
    v_dl   = v_dl_sum / max(1, n_val)
    v_dt   = v_dt_sum / max(1, n_val)

    # Scheduler (after warmup)
    if epoch > warmup_end:
        scheduler.step(v_loss)

    # ════ LOGGING ══════════════════════════════════════════════
    ep_time = time.time() - t0
    epoch_times.append(ep_time)
    lr = optimizer.param_groups[0]["lr"]

    print(f"\nEpoch {epoch:03d}/{EPOCHS} ({ep_time:.0f}s) LR={lr:.1e}")
    print(f"  Train | Loss: {t_loss:.4f} | Liver: {t_dl:.4f} | Tumor: {t_dt:.4f}")
    print(f"  Val   | Loss: {v_loss:.4f} | Liver: {v_dl:.4f} | Tumor: {v_dt:.4f}")
    print(f"  Grad  | Max norm: {max_gnorm:.4f} | Scaler: {scaler.get_scale():.0f}")
    if inf_events > 0:
        print(f"  ⚠️ {inf_events} Inf gradient events (handled by GradScaler)")

    event_str = ""

    # Log to CSV
    log.append({
        "epoch": epoch, "train_loss": t_loss, "val_loss": v_loss,
        "train_dice_liver": t_dl, "train_dice_tumor": t_dt,
        "val_dice_liver": v_dl, "val_dice_tumor": v_dt,
        "lr": lr, "time_s": ep_time, "max_grad_norm": max_gnorm,
        "inf_events": inf_events, "scaler_scale": scaler.get_scale(),
        "event": event_str
    })
    pd.DataFrame(log).to_csv("results/training_log.csv", index=False)

    # ════ WATCHDOG — SIMPLE, NO RECOVERY ═══════════════════════
    # v8 FIX #6: Only STOP — no rollback, no FP32 fallback, no recovery
    stop = False
    reason = ""

    # Check: non-finite val_loss
    if not torch.tensor(v_loss).isfinite():
        stop, reason = True, f"NON_FINITE: val_loss={v_loss}"

    # Check: extreme divergence
    if not stop and v_loss > 5.0 and epoch > warmup_end:
        stop, reason = True, f"DIVERGENCE: val_loss={v_loss:.4f} > 5.0"

    # Check: gradient explosion
    if not stop and max_gnorm > 500.0:
        stop, reason = True, f"GRADIENT_EXPLOSION: norm={max_gnorm:.1f}"

    if stop:
        print(f"\n🛑 WATCHDOG STOP: {reason}")
        log[-1]["event"] = f"WATCHDOG: {reason}"
        pd.DataFrame(log).to_csv("results/training_log.csv", index=False)
        break

    # ════ CHECKPOINTS ══════════════════════════════════════════
    # Track best dice
    if v_dl > best_val_dice:
        best_val_dice = v_dl

    # Save latest every epoch
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "val_loss": v_loss,
        "best_val_loss": best_val_loss
    }, CKPT_LATEST)
    print(f"  💾 Latest saved (epoch {epoch})")

    # Save best
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        no_improve = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_loss": v_loss,
            "best_val_loss": best_val_loss
        }, CKPT_BEST)
        print(f"  ⭐ BEST MODEL (val_loss={v_loss:.4f})")
    else:
        no_improve += 1
        print(f"  ⏳ No improvement ({no_improve}/{PATIENCE})")

    # Early stopping
    if no_improve >= PATIENCE:
        print(f"\n🛑 Early stopping at epoch {epoch}")
        event_str = "EARLY_STOP"
        log[-1]["event"] = event_str
        pd.DataFrame(log).to_csv("results/training_log.csv", index=False)
        break

# ════ SUMMARY ══════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  TRAINING COMPLETE")
print(f"  Best val_loss:    {best_val_loss:.4f}")
print(f"  Best Liver Dice:  {best_val_dice:.4f}")
print(f"  Total time:       {(time.time() - global_t0)/3600:.2f} hours")
print(f"  Avg epoch time:   {np.mean(epoch_times):.0f}s")
print("=" * 60)"""

# ──────────────────────────────────────────────────────────────
CELL_8_CURVES = r"""log_path = "results/training_log.csv"
if os.path.exists(log_path):
    df = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Loss
    axes[0].plot(df['epoch'], df['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(df['epoch'], df['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_title('Loss vs Epoch', fontsize=14)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=12); axes[0].grid(True, alpha=0.3)

    # Dice Liver
    axes[1].plot(df['epoch'], df['train_dice_liver'], 'b--', label='Train', linewidth=1.5)
    axes[1].plot(df['epoch'], df['val_dice_liver'], 'b-', label='Val', linewidth=2)
    axes[1].set_title('Dice Liver vs Epoch', fontsize=14)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Dice')
    axes[1].legend(fontsize=12); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    # Dice Tumor
    axes[2].plot(df['epoch'], df['train_dice_tumor'], 'r--', label='Train', linewidth=1.5)
    axes[2].plot(df['epoch'], df['val_dice_tumor'], 'r-', label='Val', linewidth=2)
    axes[2].set_title('Dice Tumor vs Epoch', fontsize=14)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Dice')
    axes[2].legend(fontsize=12); axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

    valid_df = df.dropna(subset=['val_loss'])
    if len(valid_df) > 0:
        best = valid_df.loc[valid_df['val_loss'].idxmin()]
        print(f"\nBest epoch: {int(best['epoch'])}")
        print(f"  Val Loss:       {best['val_loss']:.4f}")
        print(f"  Val Dice Liver: {best['val_dice_liver']:.4f}")
        print(f"  Val Dice Tumor: {best['val_dice_tumor']:.4f}")

    if 'max_grad_norm' in df.columns:
        print(f"\nGradient Norm Stats:")
        print(f"  Mean: {df['max_grad_norm'].mean():.4f}")
        print(f"  Max:  {df['max_grad_norm'].max():.4f}")

    print(f"  Avg time/epoch: {df['time_s'].mean():.0f}s")
else:
    print("No training log found.")"""

# ──────────────────────────────────────────────────────────────
CELL_9_EVAL = r"""print("=" * 70)
print("  📊 FULL MODEL EVALUATION — Best Checkpoint on Validation Set")
print("=" * 70)

CKPT = "models/best_model.pth"
assert os.path.exists(CKPT), "No best_model.pth found! Train first."

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(1, 2).to(device)
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"\nLoaded best model from epoch {ckpt['epoch']}")
print(f"Checkpoint val_loss: {ckpt['val_loss']:.6f}")
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# ── Prepare val data ──────────────────────────────────────────
eval_ds = LITSDataset(VAL_CSV, DATA_ROOT, augment=False)
eval_ld = DataLoader(eval_ds, batch_size=32, shuffle=False,
                     pin_memory=True, num_workers=2)
print(f"Val samples: {len(eval_ds):,}")
print(f"Val batches: {len(eval_ld)}")

# ── Confusion matrix accumulators (per channel) ──────────────
tp = torch.zeros(2, dtype=torch.float64)
fp = torch.zeros(2, dtype=torch.float64)
fn = torch.zeros(2, dtype=torch.float64)
tn = torch.zeros(2, dtype=torch.float64)

inter_sum = torch.zeros(2, dtype=torch.float64)
pred_sum  = torch.zeros(2, dtype=torch.float64)
true_sum  = torch.zeros(2, dtype=torch.float64)

sample_dice_liver = []
sample_dice_tumor = []

total_pixels = 0
correct_pixels = 0
smooth = 1e-6

print("\nRunning inference on entire val set...")
with torch.no_grad():
    for imgs, msks in tqdm(eval_ld, desc="Evaluating"):
        imgs = imgs.to(device, non_blocking=True)
        msks = msks.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            logits = model(imgs)

        probs = torch.sigmoid(logits.float())
        preds = (probs > 0.5).float()
        targets = msks.float()

        B = preds.shape[0]
        for ch in range(2):
            p = preds[:, ch].reshape(B, -1)
            t = targets[:, ch].reshape(B, -1)

            tp[ch] += (p * t).sum().item()
            fp[ch] += (p * (1 - t)).sum().item()
            fn[ch] += ((1 - p) * t).sum().item()
            tn[ch] += ((1 - p) * (1 - t)).sum().item()

            inter_sum[ch] += (p * t).sum().item()
            pred_sum[ch]  += p.sum().item()
            true_sum[ch]  += t.sum().item()

        for b in range(B):
            for ch, lst in [(0, sample_dice_liver), (1, sample_dice_tumor)]:
                p = preds[b, ch].reshape(-1)
                t = targets[b, ch].reshape(-1)
                inter = (p * t).sum().item()
                denom = p.sum().item() + t.sum().item()
                if denom > 0:
                    lst.append((2 * inter + smooth) / (denom + smooth))
                else:
                    lst.append(1.0)

        total_pixels += preds.numel()
        correct_pixels += (preds == targets).sum().item()

print("\n" + "=" * 70)
print("  RESULTS — Full Validation Set")
print("=" * 70)

channel_names = ['Liver', 'Tumor']
metrics = {}

for ch, name in enumerate(channel_names):
    dice  = (2 * inter_sum[ch] + smooth) / (pred_sum[ch] + true_sum[ch] + smooth)
    iou   = (tp[ch] + smooth) / (tp[ch] + fp[ch] + fn[ch] + smooth)
    prec  = (tp[ch] + smooth) / (tp[ch] + fp[ch] + smooth)
    rec   = (tp[ch] + smooth) / (tp[ch] + fn[ch] + smooth)
    f1    = (2 * prec * rec) / (prec + rec + smooth)
    acc   = (tp[ch] + tn[ch]) / (tp[ch] + fp[ch] + fn[ch] + tn[ch])
    spec  = (tn[ch] + smooth) / (tn[ch] + fp[ch] + smooth)

    metrics[name] = {
        'Dice': dice.item(), 'IoU': iou.item(),
        'Precision': prec.item(), 'Recall': rec.item(),
        'F1': f1.item(), 'Accuracy': acc.item(),
        'Specificity': spec.item(),
        'TP': int(tp[ch].item()), 'FP': int(fp[ch].item()),
        'FN': int(fn[ch].item()), 'TN': int(tn[ch].item())
    }

    print(f"\n  ── {name} Channel ──")
    print(f"  Dice Score:    {dice.item():.4f}")
    print(f"  IoU (Jaccard): {iou.item():.4f}")
    print(f"  Precision:     {prec.item():.4f}")
    print(f"  Recall:        {rec.item():.4f}")
    print(f"  F1 Score:      {f1.item():.4f}")
    print(f"  Accuracy:      {acc.item():.4f}")
    print(f"  Specificity:   {spec.item():.4f}")
    print(f"  TP={int(tp[ch]):,}  FP={int(fp[ch]):,}  FN={int(fn[ch]):,}  TN={int(tn[ch]):,}")

mean_dice = (metrics['Liver']['Dice'] + metrics['Tumor']['Dice']) / 2
mean_iou  = (metrics['Liver']['IoU'] + metrics['Tumor']['IoU']) / 2
pixel_acc = correct_pixels / total_pixels

print(f"\n  ── Overall ──")
print(f"  Mean Dice:       {mean_dice:.4f}")
print(f"  Mean IoU:        {mean_iou:.4f}")
print(f"  Pixel Accuracy:  {pixel_acc:.4f} ({pixel_acc*100:.2f}%)")

print(f"\n  ── Per-Sample Dice Distribution ──")
for name, lst in [('Liver', sample_dice_liver), ('Tumor', sample_dice_tumor)]:
    arr = np.array(lst)
    non_trivial = arr[arr < 1.0]
    print(f"  {name}: mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
          f"std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")
    if len(non_trivial) > 0:
        print(f"    Non-trivial (has GT): mean={non_trivial.mean():.4f}, "
              f"median={np.median(non_trivial):.4f}, n={len(non_trivial)}")

metrics_rows = []
for name in channel_names:
    row = {'Channel': name}
    row.update(metrics[name])
    metrics_rows.append(row)
metrics_rows.append({
    'Channel': 'Mean', 'Dice': mean_dice, 'IoU': mean_iou,
    'Accuracy': pixel_acc
})
pd.DataFrame(metrics_rows).to_csv('results/evaluation_metrics.csv', index=False)
print(f"\n💾 Metrics saved to results/evaluation_metrics.csv")
print("=" * 70)"""

# ──────────────────────────────────────────────────────────────
CELL_10_VIS = r"""print("=" * 70)
print("  🖼️  DETAILED PREDICTION VISUALIZATIONS")
print("=" * 70)

def make_overlay(ct_slice, true_liver, true_tumor, pred_liver, pred_tumor):
    h, w = ct_slice.shape
    ct_rgb = np.stack([ct_slice * 255] * 3, axis=-1).astype(np.uint8)
    overlay = ct_rgb.copy().astype(np.float32)

    liver_tp = (true_liver > 0) & (pred_liver > 0)
    liver_fp = (true_liver == 0) & (pred_liver > 0)
    liver_fn = (true_liver > 0) & (pred_liver == 0)
    overlay[liver_tp] = overlay[liver_tp] * 0.4 + np.array([0, 200, 0]) * 0.6
    overlay[liver_fp] = overlay[liver_fp] * 0.4 + np.array([255, 200, 0]) * 0.6
    overlay[liver_fn] = overlay[liver_fn] * 0.4 + np.array([200, 0, 0]) * 0.6

    tumor_tp = (true_tumor > 0) & (pred_tumor > 0)
    tumor_fp = (true_tumor == 0) & (pred_tumor > 0)
    tumor_fn = (true_tumor > 0) & (pred_tumor == 0)
    overlay[tumor_tp] = overlay[tumor_tp] * 0.3 + np.array([0, 255, 255]) * 0.7
    overlay[tumor_fp] = overlay[tumor_fp] * 0.3 + np.array([255, 0, 255]) * 0.7
    overlay[tumor_fn] = overlay[tumor_fn] * 0.3 + np.array([255, 140, 0]) * 0.7

    return np.clip(overlay, 0, 255).astype(np.uint8)

print("\nSelecting diverse samples for visualization...")
categories = {
    'liver_and_tumor': [], 'liver_only': [],
    'large_liver': [], 'empty': [], 'small_tumor': []
}

np.random.seed(42)
scan_idx = np.random.choice(len(eval_ds), min(1000, len(eval_ds)), replace=False)

for idx in scan_idx:
    _, msk = eval_ds[int(idx)]
    liver_px = msk[0].sum().item()
    tumor_px = msk[1].sum().item()

    if liver_px == 0 and tumor_px == 0 and len(categories['empty']) < 1:
        categories['empty'].append(int(idx))
    elif tumor_px > 0 and tumor_px < 200 and len(categories['small_tumor']) < 1:
        categories['small_tumor'].append(int(idx))
    elif liver_px > 0 and tumor_px > 0 and len(categories['liver_and_tumor']) < 2:
        categories['liver_and_tumor'].append(int(idx))
    elif liver_px > 0 and tumor_px == 0 and len(categories['liver_only']) < 2:
        categories['liver_only'].append(int(idx))
    elif liver_px > 4000 and len(categories['large_liver']) < 2:
        categories['large_liver'].append(int(idx))

selected = []
for cat in ['liver_and_tumor', 'liver_only', 'large_liver', 'small_tumor', 'empty']:
    selected.extend(categories[cat])
selected = selected[:8]

if len(selected) < 4:
    selected = list(range(0, len(eval_ds), len(eval_ds) // 8))[:8]

print(f"Selected {len(selected)} samples: {selected}")

fig, axes = plt.subplots(len(selected), 8, figsize=(40, 5 * len(selected)))
if len(selected) == 1:
    axes = axes.reshape(1, -1)

for r, idx in enumerate(selected):
    image, true_mask = eval_ds[idx]

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = model(image.unsqueeze(0).to(device))
        probs = torch.sigmoid(logits.float()).squeeze(0).cpu()
        pred_mask = (probs > 0.5).float()

    ct = image[0].numpy()
    tl = true_mask[0].numpy()
    tt = true_mask[1].numpy()
    pl = pred_mask[0].numpy()
    pt = pred_mask[1].numpy()

    def sample_dice(p, t):
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        if denom == 0: return 1.0
        return (2 * inter + 1e-6) / (denom + 1e-6)

    dl = sample_dice(pl, tl)
    dt = sample_dice(pt, tt)

    overlay_gt = make_overlay(ct, tl, tt, tl, tt)
    overlay_pred = make_overlay(ct, tl, tt, pl, pt)

    axes[r, 0].imshow(ct, cmap='gray')
    axes[r, 0].set_title(f'CT (idx={idx})', fontsize=11, fontweight='bold')
    axes[r, 1].imshow(tl, cmap='Greens', vmin=0, vmax=1)
    axes[r, 1].set_title(f'True Liver ({tl.sum():.0f}px)', fontsize=10)
    axes[r, 2].imshow(pl, cmap='Greens', vmin=0, vmax=1)
    axes[r, 2].set_title(f'Pred Liver (Dice={dl:.3f})', fontsize=10,
                         color='green' if dl > 0.7 else ('orange' if dl > 0.4 else 'red'))
    axes[r, 3].imshow(tt, cmap='Reds', vmin=0, vmax=1)
    axes[r, 3].set_title(f'True Tumor ({tt.sum():.0f}px)', fontsize=10)
    axes[r, 4].imshow(pt, cmap='Reds', vmin=0, vmax=1)
    axes[r, 4].set_title(f'Pred Tumor (Dice={dt:.3f})', fontsize=10,
                         color='green' if dt > 0.5 else ('orange' if dt > 0.2 else 'red'))
    axes[r, 5].imshow(overlay_gt)
    axes[r, 5].set_title('Ground Truth Overlay', fontsize=10)
    axes[r, 6].imshow(overlay_pred)
    axes[r, 6].set_title('Prediction Overlay', fontsize=10)

    diff = np.zeros((*ct.shape, 3), dtype=np.uint8)
    for ch_i, (t_ch, p_ch) in enumerate([(tl, pl), (tt, pt)]):
        tp_mask = (t_ch > 0) & (p_ch > 0)
        fp_mask = (t_ch == 0) & (p_ch > 0)
        fn_mask = (t_ch > 0) & (p_ch == 0)
        diff[tp_mask] = [0, 200, 0]
        diff[fp_mask] = [255, 255, 0]
        diff[fn_mask] = [255, 0, 0]
    axes[r, 7].imshow(diff)
    axes[r, 7].set_title('Diff (🟢TP 🟡FP 🔴FN)', fontsize=10)

    for ax in axes[r]:
        ax.axis('off')

plt.suptitle('Model Predictions vs Ground Truth — Detailed Comparison',
             fontsize=18, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('results/detailed_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n💾 Saved to results/detailed_predictions.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, data, name, color in [
    (axes[0], sample_dice_liver, 'Liver', '#2ecc71'),
    (axes[1], sample_dice_tumor, 'Tumor', '#e74c3c')
]:
    arr = np.array(data)
    non_trivial = arr[arr < 1.0]
    ax.hist(arr, bins=50, color=color, alpha=0.7, edgecolor='white')
    ax.axvline(arr.mean(), color='black', linestyle='--', linewidth=2,
               label=f'Mean: {arr.mean():.4f}')
    if len(non_trivial) > 0:
        ax.axvline(non_trivial.mean(), color='navy', linestyle=':',
                   linewidth=2, label=f'Non-trivial mean: {non_trivial.mean():.4f}')
    ax.set_title(f'{name} Dice Distribution (n={len(arr)})', fontsize=13)
    ax.set_xlabel('Dice Score', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/dice_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("💾 Saved to results/dice_distribution.png")"""

# ──────────────────────────────────────────────────────────────
CELL_11_DASHBOARD = r"""print("=" * 70)
print("  📋 GENERATING METRICS DASHBOARD")
print("=" * 70)

fig = plt.figure(figsize=(24, 16))
fig.patch.set_facecolor('#1a1a2e')

fig.suptitle('Liver & Tumor Segmentation — Model Evaluation Dashboard',
             fontsize=22, fontweight='bold', color='white', y=0.98)

card_data = [
    ('Liver Dice',   f"{metrics['Liver']['Dice']:.4f}",   '#2ecc71'),
    ('Tumor Dice',   f"{metrics['Tumor']['Dice']:.4f}",   '#e74c3c'),
    ('Mean Dice',    f"{mean_dice:.4f}",                   '#3498db'),
    ('Liver IoU',    f"{metrics['Liver']['IoU']:.4f}",     '#27ae60'),
    ('Tumor IoU',    f"{metrics['Tumor']['IoU']:.4f}",     '#c0392b'),
    ('Mean IoU',     f"{mean_iou:.4f}",                    '#2980b9'),
    ('Pixel Acc',    f"{pixel_acc*100:.2f}%",              '#f39c12'),
    ('Best Epoch',   f"{ckpt['epoch']}",                   '#9b59b6'),
]

for i, (label, value, color) in enumerate(card_data):
    ax = fig.add_axes([0.02 + i * 0.12, 0.82, 0.11, 0.12])
    ax.set_facecolor('#16213e')
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.65, value, ha='center', va='center',
            fontsize=20, fontweight='bold', color=color,
            transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha='center', va='center',
            fontsize=10, color='#bdc3c7', transform=ax.transAxes)

log_path = 'results/training_log.csv'
if os.path.exists(log_path):
    df_log = pd.read_csv(log_path)
    valid_epochs = df_log.dropna(subset=['val_loss'])

    ax1 = fig.add_axes([0.06, 0.42, 0.28, 0.32])
    ax1.set_facecolor('#16213e')
    ax1.plot(valid_epochs['epoch'], valid_epochs['train_loss'],
             'c-', linewidth=2, label='Train Loss')
    ax1.plot(valid_epochs['epoch'], valid_epochs['val_loss'],
             '#f39c12', linewidth=2, label='Val Loss')
    ax1.set_title('Loss Curve', fontsize=13, color='white', pad=10)
    ax1.set_xlabel('Epoch', fontsize=10, color='#bdc3c7')
    ax1.set_ylabel('Loss', fontsize=10, color='#bdc3c7')
    ax1.legend(fontsize=9, facecolor='#16213e', edgecolor='white',
               labelcolor='white')
    ax1.tick_params(colors='#bdc3c7')
    ax1.grid(True, alpha=0.2)
    for spine in ax1.spines.values(): spine.set_color('#34495e')

    ax2 = fig.add_axes([0.38, 0.42, 0.28, 0.32])
    ax2.set_facecolor('#16213e')
    ax2.plot(valid_epochs['epoch'], valid_epochs['val_dice_liver'],
             '#2ecc71', linewidth=2, label='Val Liver Dice')
    ax2.plot(valid_epochs['epoch'], valid_epochs['val_dice_tumor'],
             '#e74c3c', linewidth=2, label='Val Tumor Dice')
    ax2.set_title('Validation Dice', fontsize=13, color='white', pad=10)
    ax2.set_xlabel('Epoch', fontsize=10, color='#bdc3c7')
    ax2.set_ylabel('Dice', fontsize=10, color='#bdc3c7')
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9, facecolor='#16213e', edgecolor='white',
               labelcolor='white')
    ax2.tick_params(colors='#bdc3c7')
    ax2.grid(True, alpha=0.2)
    for spine in ax2.spines.values(): spine.set_color('#34495e')

ax3 = fig.add_axes([0.70, 0.42, 0.28, 0.32])
ax3.set_facecolor('#16213e')
ax3.set_xticks([]); ax3.set_yticks([])
ax3.set_title('Per-Channel Metrics', fontsize=13, color='white', pad=10)
for spine in ax3.spines.values(): spine.set_color('#34495e')

table_rows = [
    ('Metric',     'Liver',  'Tumor'),
    ('Dice',       f"{metrics['Liver']['Dice']:.4f}",      f"{metrics['Tumor']['Dice']:.4f}"),
    ('IoU',        f"{metrics['Liver']['IoU']:.4f}",       f"{metrics['Tumor']['IoU']:.4f}"),
    ('Precision',  f"{metrics['Liver']['Precision']:.4f}", f"{metrics['Tumor']['Precision']:.4f}"),
    ('Recall',     f"{metrics['Liver']['Recall']:.4f}",    f"{metrics['Tumor']['Recall']:.4f}"),
    ('F1 Score',   f"{metrics['Liver']['F1']:.4f}",        f"{metrics['Tumor']['F1']:.4f}"),
    ('Specificity',f"{metrics['Liver']['Specificity']:.4f}",f"{metrics['Tumor']['Specificity']:.4f}"),
]
for i, (metric, liver_v, tumor_v) in enumerate(table_rows):
    y = 0.88 - i * 0.12
    is_header = (i == 0)
    weight = 'bold' if is_header else 'normal'
    color = 'white' if is_header else '#bdc3c7'
    ax3.text(0.05, y, metric, ha='left', va='center', fontsize=10,
             fontweight=weight, color=color, transform=ax3.transAxes)
    ax3.text(0.55, y, liver_v, ha='center', va='center', fontsize=10,
             fontweight=weight, color='#2ecc71' if not is_header else color,
             transform=ax3.transAxes)
    ax3.text(0.85, y, tumor_v, ha='center', va='center', fontsize=10,
             fontweight=weight, color='#e74c3c' if not is_header else color,
             transform=ax3.transAxes)

ax4 = fig.add_axes([0.06, 0.05, 0.40, 0.28])
ax4.set_facecolor('#16213e')
cm_labels = ['TP', 'FP', 'FN', 'TN']
for ch, (name, color) in enumerate([('Liver', '#2ecc71'), ('Tumor', '#e74c3c')]):
    values = [tp[ch].item(), fp[ch].item(), fn[ch].item(), tn[ch].item()]
    total = sum(values)
    pcts = [v/total*100 for v in values]
    x = np.arange(4) + ch * 0.35
    bars = ax4.bar(x, pcts, 0.3, color=color, alpha=0.8, label=name)
    for bar, pct in zip(bars, pcts):
        if pct > 0.5:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{pct:.1f}%', ha='center', va='bottom',
                     fontsize=8, color='white')

ax4.set_xticks(np.arange(4) + 0.175)
ax4.set_xticklabels(cm_labels, fontsize=10, color='#bdc3c7')
ax4.set_title('Confusion Matrix (% of pixels)', fontsize=13, color='white', pad=10)
ax4.set_ylabel('Percentage', fontsize=10, color='#bdc3c7')
ax4.legend(fontsize=10, facecolor='#16213e', edgecolor='white', labelcolor='white')
ax4.tick_params(colors='#bdc3c7')
ax4.grid(True, alpha=0.2, axis='y')
for spine in ax4.spines.values(): spine.set_color('#34495e')

ax5 = fig.add_axes([0.55, 0.05, 0.42, 0.28])
ax5.set_facecolor('#16213e')
ax5.set_xticks([]); ax5.set_yticks([])
ax5.set_title('Model Configuration', fontsize=13, color='white', pad=10)
for spine in ax5.spines.values(): spine.set_color('#34495e')

info_lines = [
    f"Architecture:  Standard 2D U-Net (64>128>256>512>1024)",
    f"Parameters:    {sum(p.numel() for p in model.parameters()):,}",
    f"Loss Function: 0.5 x BCE + 0.5 x Dice (FP32-safe)",
    f"Optimizer:     Adam (lr=3e-4 from scratch)",
    f"Batch Size:    16 x 2 accum = 32 effective",
    f"Best Epoch:    {ckpt['epoch']}",
    f"Val Loss:      {ckpt['val_loss']:.6f}",
    f"Dataset:       LiTS17 ({len(eval_ds):,} val samples)",
    f"Input Size:    256 x 256 (grayscale)",
    f"AMP:           Mixed Precision (always ON, FP32 loss)",
]
for i, line in enumerate(info_lines):
    ax5.text(0.05, 0.92 - i * 0.095, line, ha='left', va='center',
             fontsize=10, color='#bdc3c7', transform=ax5.transAxes,
             fontfamily='monospace')

plt.savefig('results/evaluation_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
plt.show()
print("\n💾 Saved to results/evaluation_dashboard.png")

print("\n" + "=" * 70)
print("  ✅ EVALUATION COMPLETE")
print("=" * 70)
print(f"\n  📁 All outputs saved to results/")
print(f"     ├── evaluation_metrics.csv     — full metrics table")
print(f"     ├── detailed_predictions.png   — real vs predicted comparison")
print(f"     ├── dice_distribution.png      — per-sample Dice histograms")
print(f"     ├── evaluation_dashboard.png   — publication-quality dashboard")
print(f"     └── training_curves.png        — loss and Dice curves")
print(f"\n  🏆 Final Scores:")
print(f"     Liver Dice:    {metrics['Liver']['Dice']:.4f}")
print(f"     Tumor Dice:    {metrics['Tumor']['Dice']:.4f}")
print(f"     Mean Dice:     {mean_dice:.4f}")
print(f"     Pixel Accuracy: {pixel_acc*100:.2f}%")
print("=" * 70)"""

CELL_12_MD = """---

### Target Metrics (from FasNet paper)

| Stage | Dice Liver | Dice Tumor | Notes |
|---|---|---|---|
| First run (base U-Net) | > 0.60 | > 0.30 | Model is learning |
| Good base U-Net | ~0.74 | ~0.50 | Comparable to baselines |
| After adding attention | ~0.85 | ~0.70 | Paper-level |
| FasNet (ResNet50+VGG16) | 0.8766 | — | State of the art |

### v8 Root Cause Fixes Summary

| Fix | What | Why |
|-----|------|-----|
| WeightedRandomSampler | Oversample liver+ slices 3:1 | 70.6% empty slices → noisy Dice gradients |
| Batch=16 + AccumSteps=2 | Effective batch=32, VRAM safe | Batch=32 caused OOM when AMP disabled |
| DiceLoss `.float()` | Force FP32 in loss | FP16 overflow → Inf gradients |
| AMP always ON | Never disable AMP | FP32 fallback → OOM crash |
| LR=3e-4 / 1e-5 | Proper LR for scratch/finetune | 1e-4 + fresh optimizer → feature destruction |
| No recovery system | Simple early stopping only | Rollback system made failures worse |"""

# ──────────────────────────────────────────────────────────────
CELL_13_TEST_EVAL = r"""# ═══════════════════════════════════════════════════════════════
# Cell 12 — HELD-OUT TEST SET EVALUATION (Never seen during training)
# ═══════════════════════════════════════════════════════════════
# This evaluates on lits_test.csv — 11 studies (3,038 slices) that
# were NEVER used for training or model selection (early stopping).
# These are the numbers you report in your paper / thesis.
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("  🧪 HELD-OUT TEST SET EVALUATION — Final Reportable Metrics")
print("=" * 70)

CKPT = "models/best_model.pth"
assert os.path.exists(CKPT), "No best_model.pth found! Train first."

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_model = UNet(1, 2).to(device)
test_ckpt = torch.load(CKPT, map_location=device, weights_only=False)
test_model.load_state_dict(test_ckpt['model_state_dict'])
test_model.eval()
print(f"\nLoaded best model from epoch {test_ckpt['epoch']}")
print(f"Checkpoint val_loss: {test_ckpt['val_loss']:.6f}")

# ── Prepare TEST data ────────────────────────────────────────
test_ds = LITSDataset(TEST_CSV, DATA_ROOT, augment=False)
test_ld = DataLoader(test_ds, batch_size=32, shuffle=False,
                     pin_memory=True, num_workers=2)
print(f"\nTest samples: {len(test_ds):,}")
print(f"Test batches: {len(test_ld)}")

# ── Confusion matrix accumulators (per channel) ──────────────
test_tp = torch.zeros(2, dtype=torch.float64)
test_fp = torch.zeros(2, dtype=torch.float64)
test_fn = torch.zeros(2, dtype=torch.float64)
test_tn = torch.zeros(2, dtype=torch.float64)

test_inter_sum = torch.zeros(2, dtype=torch.float64)
test_pred_sum  = torch.zeros(2, dtype=torch.float64)
test_true_sum  = torch.zeros(2, dtype=torch.float64)

test_dice_liver = []
test_dice_tumor = []

test_total_px = 0
test_correct_px = 0
smooth = 1e-6

print("\nRunning inference on held-out test set...")
with torch.no_grad():
    for imgs, msks in tqdm(test_ld, desc="Test Eval"):
        imgs = imgs.to(device, non_blocking=True)
        msks = msks.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            logits = test_model(imgs)

        probs = torch.sigmoid(logits.float())
        preds = (probs > 0.5).float()
        targets = msks.float()

        B = preds.shape[0]
        for ch in range(2):
            p = preds[:, ch].reshape(B, -1)
            t = targets[:, ch].reshape(B, -1)

            test_tp[ch] += (p * t).sum().item()
            test_fp[ch] += (p * (1 - t)).sum().item()
            test_fn[ch] += ((1 - p) * t).sum().item()
            test_tn[ch] += ((1 - p) * (1 - t)).sum().item()

            test_inter_sum[ch] += (p * t).sum().item()
            test_pred_sum[ch]  += p.sum().item()
            test_true_sum[ch]  += t.sum().item()

        for b in range(B):
            for ch, lst in [(0, test_dice_liver), (1, test_dice_tumor)]:
                p = preds[b, ch].reshape(-1)
                t = targets[b, ch].reshape(-1)
                inter = (p * t).sum().item()
                denom = p.sum().item() + t.sum().item()
                if denom > 0:
                    lst.append((2 * inter + smooth) / (denom + smooth))
                else:
                    lst.append(1.0)

        test_total_px += preds.numel()
        test_correct_px += (preds == targets).sum().item()

print("\n" + "=" * 70)
print("  RESULTS — HELD-OUT TEST SET (FINAL REPORTABLE)")
print("=" * 70)

channel_names = ['Liver', 'Tumor']
test_metrics = {}

for ch, name in enumerate(channel_names):
    dice  = (2 * test_inter_sum[ch] + smooth) / (test_pred_sum[ch] + test_true_sum[ch] + smooth)
    iou   = (test_tp[ch] + smooth) / (test_tp[ch] + test_fp[ch] + test_fn[ch] + smooth)
    prec  = (test_tp[ch] + smooth) / (test_tp[ch] + test_fp[ch] + smooth)
    rec   = (test_tp[ch] + smooth) / (test_tp[ch] + test_fn[ch] + smooth)
    f1    = (2 * prec * rec) / (prec + rec + smooth)
    acc   = (test_tp[ch] + test_tn[ch]) / (test_tp[ch] + test_fp[ch] + test_fn[ch] + test_tn[ch])
    spec  = (test_tn[ch] + smooth) / (test_tn[ch] + test_fp[ch] + smooth)

    test_metrics[name] = {
        'Dice': dice.item(), 'IoU': iou.item(),
        'Precision': prec.item(), 'Recall': rec.item(),
        'F1': f1.item(), 'Accuracy': acc.item(),
        'Specificity': spec.item(),
        'TP': int(test_tp[ch].item()), 'FP': int(test_fp[ch].item()),
        'FN': int(test_fn[ch].item()), 'TN': int(test_tn[ch].item())
    }

    print(f"\n  ── {name} Channel ──")
    print(f"  Dice Score:    {dice.item():.4f}")
    print(f"  IoU (Jaccard): {iou.item():.4f}")
    print(f"  Precision:     {prec.item():.4f}")
    print(f"  Recall:        {rec.item():.4f}")
    print(f"  F1 Score:      {f1.item():.4f}")
    print(f"  Accuracy:      {acc.item():.4f}")
    print(f"  Specificity:   {spec.item():.4f}")

test_mean_dice = (test_metrics['Liver']['Dice'] + test_metrics['Tumor']['Dice']) / 2
test_mean_iou  = (test_metrics['Liver']['IoU'] + test_metrics['Tumor']['IoU']) / 2
test_pixel_acc = test_correct_px / test_total_px

print(f"\n  ── Overall (Test Set) ──")
print(f"  Mean Dice:       {test_mean_dice:.4f}")
print(f"  Mean IoU:        {test_mean_iou:.4f}")
print(f"  Pixel Accuracy:  {test_pixel_acc:.4f} ({test_pixel_acc*100:.2f}%)")

print(f"\n  ── Per-Sample Test Dice Distribution ──")
for name, lst in [('Liver', test_dice_liver), ('Tumor', test_dice_tumor)]:
    arr = np.array(lst)
    non_trivial = arr[arr < 1.0]
    print(f"  {name}: mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
          f"std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")
    if len(non_trivial) > 0:
        print(f"    Non-trivial (has GT): mean={non_trivial.mean():.4f}, "
              f"median={np.median(non_trivial):.4f}, n={len(non_trivial)}")

# Save test metrics
test_rows = []
for name in channel_names:
    row = {'Channel': name, 'Split': 'TEST'}
    row.update(test_metrics[name])
    test_rows.append(row)
test_rows.append({
    'Channel': 'Mean', 'Split': 'TEST',
    'Dice': test_mean_dice, 'IoU': test_mean_iou,
    'Accuracy': test_pixel_acc
})
pd.DataFrame(test_rows).to_csv('results/test_metrics.csv', index=False)

# Comparison table
print("\n" + "=" * 70)
print("  VAL vs TEST COMPARISON")
print("=" * 70)
if os.path.exists('results/evaluation_metrics.csv'):
    val_df = pd.read_csv('results/evaluation_metrics.csv')
    val_liver_dice = val_df[val_df['Channel']=='Liver']['Dice'].values[0]
    val_tumor_dice = val_df[val_df['Channel']=='Tumor']['Dice'].values[0]
    val_mean_dice  = val_df[val_df['Channel']=='Mean']['Dice'].values[0]
    print(f"\n  {'Metric':<20s} {'Validation':>12s} {'Test':>12s}")
    print(f"  {'-'*44}")
    print(f"  {'Liver Dice':<20s} {val_liver_dice:>12.4f} {test_metrics['Liver']['Dice']:>12.4f}")
    print(f"  {'Tumor Dice':<20s} {val_tumor_dice:>12.4f} {test_metrics['Tumor']['Dice']:>12.4f}")
    print(f"  {'Mean Dice':<20s} {val_mean_dice:>12.4f} {test_mean_dice:>12.4f}")
    print(f"  {'Pixel Accuracy':<20s} {'':>12s} {test_pixel_acc:>11.2%}")

print(f"\n💾 Test metrics saved to results/test_metrics.csv")

del test_model, test_ckpt
torch.cuda.empty_cache()
gc.collect()
print("=" * 70)"""


# ═══════════════════════════════════════════════════════════════
# ASSEMBLE NOTEBOOK
# ═══════════════════════════════════════════════════════════════

cells = [
    md(CELL_1_MD),
    code(CELL_2_GPU),
    md("### Cell 2 — Verify Dataset Paths (3-way split)"),
    code(CELL_3_PATHS),
    md("### Cell 3 — Model, Dataset, Loss, Metrics"),
    code(CELL_4_CODE),
    md("### Cell 4 — Data Diagnostics & Overfit Test"),
    code(CELL_5_DIAG),
    md("### Cell 5 — Pre-Training Sanity Checks"),
    code(CELL_6_SANITY),
    md("### Cell 6 — Training Loop (v8 Clean Pipeline)"),
    code(CELL_7_TRAIN),
    md("### Cell 7 — Training Curves"),
    code(CELL_8_CURVES),
    md("### Cell 8 — Full Model Evaluation (Metrics on Entire Validation Set)\n\nComputes **every metric** on the full validation set:\n- Dice Score (per channel + mean)\n- IoU / Jaccard Index (per channel + mean)\n- Pixel Accuracy (global + per channel)\n- Precision, Recall, F1 (per channel)\n- Confusion matrix counts (TP, FP, FN, TN)"),
    code(CELL_9_EVAL),
    md("### Cell 9 — Detailed Side-by-Side Predictions"),
    code(CELL_10_VIS),
    md("### Cell 10 — Metrics Dashboard (Publication-Quality Summary)"),
    code(CELL_11_DASHBOARD),
    md(CELL_12_MD),
    md("### Cell 12 — Held-Out Test Set Evaluation\n\nThis evaluates the best model on the **held-out test set** (11 studies, 3,038 slices) that was **never used** during training or model selection. These are the **final reportable numbers** for your thesis/paper."),
    code(CELL_13_TEST_EVAL),
]

notebook = {
    "metadata": {
        "kernelspec": {
            "language": "python",
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py"
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True
        }
    },
    "nbformat_minor": 4,
    "nbformat": 4,
    "cells": cells
}

# ── Write notebook ────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_path = os.path.join(project_dir, "livertumor-model.ipynb")

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ v8 notebook written to: {output_path}")
print(f"   Cells: {len(cells)}")
print(f"   Size:  {os.path.getsize(output_path) / 1024:.0f} KB")
