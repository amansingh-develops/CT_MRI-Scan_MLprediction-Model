# PRD: Liver Disease Detection AI System
## (Liver Tumor Segmentation using U-Net on LiTS Dataset)

---

## 0. HOW TO USE THIS PRD


> Read every section before writing any code.
> Follow the order: EDA → Model → Training → Platform.
> Never skip a phase. Each phase depends on the previous one.

---

## 1. PROJECT OVERVIEW

**Goal:** Build an end-to-end AI system that takes CT scan images as input and segments the liver and tumor regions using deep learning (U-Net architecture).

**Dataset:** LiTS (Liver Tumor Segmentation Challenge)
**Model:** U-Net (encoder-decoder with skip connections)
**Framework:** PyTorch
**Training:** Cloud GPU (Kaggle / Google Colab) — NOT local machine
**Local Machine Role:** Code writing, EDA, file management only (low disk space)

---

## 2. CONSTRAINTS & RULES

| Constraint | Detail |
|---|---|
| Local disk space | LOW — do not download full dataset locally |
| Training | Always on cloud GPU (Kaggle/Colab) |
| Dataset | Already split via CSV files — use them |
| Input shape | `[B, 1, 256, 256]` — grayscale CT slices |
| Output shape | `[B, 2, 256, 256]` — 2-channel mask (liver + tumor) |
| DataLoader | Already built and working — do not rebuild from scratch |
| Preprocessing | Done dynamically inside Dataset class — no pre-saved files |
| Framework | PyTorch only — no TensorFlow, no Keras |

---

## 3. PROJECT FOLDER STRUCTURE

```
liver_ai_project/
├── data/
│   ├── dataset_6/                  # 175k CT slices (on cloud only)
│   ├── lits_train.csv
│   ├── lits_test.csv
│   ├── lits_probe.csv
│   └── lits_df.csv
│
├── src/
│   ├── data/
│   │   ├── dataset_loader.py       # DONE — PyTorch Dataset class
│   │   └── test_loader.py          # DONE — loader test script
│   │
│   ├── eda/
│   │   ├── eda_basic.py            # Phase 1 — to be built
│   │   └── eda_visual.py           # Phase 1 — to be built
│   │
│   ├── models/
│   │   └── unet.py                 # Phase 2 — to be built
│   │
│   ├── training/
│   │   ├── loss.py                 # Phase 3 — Dice + BCE loss
│   │   ├── train.py                # Phase 3 — training loop
│   │   ├── validate.py             # Phase 3 — validation loop
│   │   └── metrics.py              # Phase 3 — Dice score metric
│   │
│   └── utils/
│       ├── visualize.py            # helpers for plotting
│       └── checkpoint.py           # save/load model weights
│
├── notebooks/
│   ├── 01_eda.ipynb                # EDA notebook for Kaggle/Colab
│   └── 02_train.ipynb              # Training notebook for Kaggle/Colab
│
├── models/                         # saved .pth checkpoint files
├── results/                        # plots, metrics, predictions
├── requirements.txt
└── README.md
```

---

## 4. PHASES (BUILD ORDER)

### PHASE 1 — EDA (Exploratory Data Analysis)
### PHASE 2 — U-Net Model Architecture
### PHASE 3 — Loss Functions + Metrics
### PHASE 4 — Training Loop
### PHASE 5 — Cloud Notebook Setup
### PHASE 6 — Platform / Inference (LAST — after model works)

> DO NOT jump to Phase 6 until Phase 4 is complete and model trains successfully.

---

## 5. PHASE 1 — EDA

**File:** `src/eda/eda_basic.py` and `src/eda/eda_visual.py`
**Notebook:** `notebooks/01_eda.ipynb`

### 5.1 Purpose
Understand the dataset before training. Find problems early.

### 5.2 What EDA Must Do

#### CSV Analysis (`eda_basic.py`)
```
- Load all 4 CSVs: lits_train, lits_test, lits_probe, lits_df
- Print: total rows, columns, dtypes
- Count: how many slices have liver (liver_mask_empty == False)
- Count: how many slices have tumor (tumor_mask_empty == False)
- Count: slices with BOTH liver and tumor
- Count: slices with NEITHER (background only)
- Print per-patient slice count (group by study_number)
- Check for missing/null values
- Print class balance as percentages
```

Expected output format:
```
Total slices: 175,000
Liver present: 82,000 (46.8%)
Tumor present: 14,000 (8.0%)
Both present: 12,000 (6.8%)
Neither: 93,000 (53.2%)
```

#### Visual Analysis (`eda_visual.py`)
```
- Pick 5 random slices from train CSV
- For each: load CT image + liver mask + tumor mask
- Display as 3-panel plot: [CT | Liver Mask | Tumor Mask]
- Save plots to results/eda/
- Also overlay tumor mask on CT image (colored heatmap)
- Plot histogram of pixel intensity values from 10 random images
- Plot bar chart: slice type distribution (liver only / tumor / neither)
```

### 5.3 EDA Rules
- Use matplotlib only (no seaborn dependency needed)
- Save all figures — do not just plt.show()
- Use the existing DataLoader — do not re-implement image loading
- EDA should run in under 2 minutes on CPU

---

## 6. PHASE 2 — U-NET MODEL

**File:** `src/models/unet.py`

### 6.1 What is U-Net (for the AI agent)

U-Net is an encoder-decoder neural network with skip connections.

```
INPUT IMAGE [1, 256, 256]
       ↓
[ENCODER — goes down, extracts features, reduces spatial size]
  Block 1: Conv → Conv → MaxPool  → feature map: 64 ch, 128x128
  Block 2: Conv → Conv → MaxPool  → feature map: 128 ch, 64x64
  Block 3: Conv → Conv → MaxPool  → feature map: 256 ch, 32x32
  Block 4: Conv → Conv → MaxPool  → feature map: 512 ch, 16x16
       ↓
[BOTTLENECK — deepest point, no spatial reduction]
  Conv → Conv                     → feature map: 1024 ch, 16x16
       ↓
[DECODER — goes up, recovers spatial size using skip connections]
  Block 4 up: Upsample + concat(skip4) → 512 ch, 32x32
  Block 3 up: Upsample + concat(skip3) → 256 ch, 64x64
  Block 2 up: Upsample + concat(skip2) → 128 ch, 128x128
  Block 1 up: Upsample + concat(skip1) → 64 ch, 256x256
       ↓
[OUTPUT HEAD]
  1x1 Conv → 2 channels (liver channel, tumor channel)
OUTPUT MASK [2, 256, 256]
```

### 6.2 Architecture Spec

#### DoubleConv Block
```python
class DoubleConv(nn.Module):
    """
    Two consecutive: Conv2d → BatchNorm2d → ReLU
    in_channels: number of input feature channels
    out_channels: number of output feature channels
    kernel_size: 3x3 (always)
    padding: 1 (to keep spatial size the same)
    """
```

#### EncoderBlock
```python
class EncoderBlock(nn.Module):
    """
    DoubleConv → save output as skip connection → MaxPool2d(2)
    Returns: (pooled_output, skip_connection)
    """
```

#### DecoderBlock
```python
class DecoderBlock(nn.Module):
    """
    ConvTranspose2d(stride=2) to upsample
    Concatenate with skip connection from encoder
    DoubleConv to process merged features
    
    IMPORTANT: Handle size mismatch — if skip and upsampled differ by 1 pixel,
    use F.interpolate or crop to match before concat.
    """
```

#### UNet (main class)
```python
class UNet(nn.Module):
    """
    in_channels: 1 (grayscale CT)
    out_channels: 2 (liver mask + tumor mask)
    features: [64, 128, 256, 512] — number of filters at each level
    
    forward(x) returns: logits of shape [B, 2, H, W]
    NO sigmoid in forward — apply during loss/inference only
    """
```

### 6.3 Model Rules
- Use `nn.BatchNorm2d` after every Conv layer
- Use `ReLU` activation (not LeakyReLU, not ELU)
- Use `ConvTranspose2d` for upsampling (not bilinear — teach clearly)
- No dropout in base version
- Final layer: `nn.Conv2d(64, out_channels=2, kernel_size=1)` — 1x1 conv
- No sigmoid in forward pass
- Model must work with input `[B, 1, 256, 256]` and output `[B, 2, 256, 256]`

### 6.4 Sanity Test (add to unet.py bottom)
```python
if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=2)
    x = torch.randn(4, 1, 256, 256)
    out = model(x)
    print(f"Input: {x.shape}")   # [4, 1, 256, 256]
    print(f"Output: {out.shape}")  # [4, 2, 256, 256]
    print("U-Net OK")
```

---

## 7. PHASE 3 — LOSS FUNCTIONS + METRICS

**File:** `src/training/loss.py` and `src/training/metrics.py`

### 7.1 Loss Functions

#### Dice Loss
```
Purpose: Measures overlap between predicted mask and ground truth mask.
Range: 0 (perfect) to 1 (no overlap).
Formula: Dice = 1 - (2 * intersection) / (sum_of_both + epsilon)
epsilon = 1e-6  ← prevents division by zero

Apply sigmoid to model output BEFORE computing Dice Loss.
Compute separately for liver channel and tumor channel, then average.
```

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        ...
    def forward(self, pred, target):
        # pred: [B, 2, H, W] raw logits
        # target: [B, 2, H, W] binary masks (0 or 1)
        # Apply sigmoid, compute per-channel dice, return mean
```

#### BCE Loss
```
Purpose: Binary Cross Entropy — standard loss for binary classification per pixel.
Use: nn.BCEWithLogitsLoss() — takes raw logits (no sigmoid needed beforehand).
```

#### Combined Loss
```python
class CombinedLoss(nn.Module):
    """
    total_loss = 0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss
    
    Why combine?
    - BCE is good at learning from individual pixels
    - Dice is good at handling class imbalance (many background pixels)
    - Together they train more stably
    """
```

### 7.2 Metrics

#### Dice Score (for evaluation, not training)
```python
def dice_score(pred, target, threshold=0.5):
    """
    pred: raw logits [B, 2, H, W]
    target: binary mask [B, 2, H, W]
    
    1. Apply sigmoid to pred
    2. Threshold at 0.5 → binary prediction
    3. Compute Dice for liver channel
    4. Compute Dice for tumor channel
    5. Return: (dice_liver, dice_tumor, dice_mean)
    """
```

---

## 8. PHASE 4 — TRAINING LOOP

**Files:** `src/training/train.py`, `src/training/validate.py`

### 8.1 Training Config (hardcoded defaults, not argparse)

```python
CONFIG = {
    "train_csv": "data/lits_train.csv",
    "val_csv": "data/lits_probe.csv",     # use probe as validation
    "image_size": 256,
    "batch_size": 8,                       # start small for safety
    "num_epochs": 30,
    "learning_rate": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "models/",
    "results_dir": "results/",
    "save_every_n_epochs": 5,
    "early_stopping_patience": 7,          # stop if no improvement for 7 epochs
}
```

### 8.2 Training Loop Structure

```
FOR each epoch:
    SET model to train mode
    FOR each batch in train_loader:
        1. Move image + mask to device (GPU)
        2. Forward pass: pred = model(image)
        3. Compute loss: combined_loss(pred, mask)
        4. Zero gradients: optimizer.zero_grad()
        5. Backward pass: loss.backward()
        6. Optimizer step: optimizer.step()
        7. Accumulate batch loss and dice score
    
    COMPUTE epoch train loss and train dice
    
    SET model to eval mode
    WITH torch.no_grad():
        FOR each batch in val_loader:
            Forward pass only
            Compute val loss and val dice
    
    COMPUTE epoch val loss and val dice
    
    PRINT: Epoch N | Train Loss | Val Loss | Train Dice | Val Dice
    SAVE metrics to results/training_log.csv
    
    IF val_loss improved: save checkpoint
    IF no improvement for patience epochs: stop training
```

### 8.3 Optimizer + Scheduler

```python
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)
# Reduce LR by half if val_loss doesn't improve for 3 epochs
```

### 8.4 Checkpoint Management

```python
# src/utils/checkpoint.py

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    """Save model state, optimizer state, epoch, loss"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    """Load checkpoint. Return epoch and val_loss."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss']
```

### 8.5 Training Log

Save a CSV file to `results/training_log.csv` with columns:
```
epoch, train_loss, val_loss, train_dice_liver, train_dice_tumor, val_dice_liver, val_dice_tumor, lr
```

### 8.6 Avoiding Overfitting

Include these in training loop:
- Early stopping (patience = 7 epochs)
- LR scheduler (ReduceLROnPlateau)
- Save only best model (based on val_loss)
- Log train vs val loss every epoch (so user can spot divergence)

DO NOT add dropout or augmentation in Phase 4. Keep it simple first.

---

## 9. PHASE 5 — CLOUD NOTEBOOK

**File:** `notebooks/02_train.ipynb`

This notebook runs on Kaggle or Google Colab (GPU runtime).

### 9.1 Notebook Cell Order

```
Cell 1: Install dependencies
  !pip install torch torchvision pandas Pillow matplotlib

Cell 2: Mount Google Drive (if Colab) OR check Kaggle dataset path

Cell 3: Clone or upload project src/ files

Cell 4: Set CONFIG paths to cloud dataset location

Cell 5: Import all modules (dataset_loader, unet, loss, train)

Cell 6: Sanity check — run model on 1 batch, print shapes

Cell 7: Create DataLoaders (train + val)

Cell 8: Initialize model, optimizer, loss

Cell 9: Run training loop (call train.py functions)

Cell 10: Plot training curves (loss and dice vs epoch)

Cell 11: Save model to Drive / Kaggle output
```

### 9.2 Cloud-Specific Notes

```
- Kaggle: Dataset is at /kaggle/input/lits-dataset/
- Colab: Mount Drive, dataset at /content/drive/MyDrive/lits/
- Always check GPU: print(torch.cuda.is_available())
- Use batch_size=16 on Kaggle (T4 GPU has 16GB)
- Use pin_memory=True and num_workers=2 in DataLoader for speed
```

---

## 10. PHASE 6 — PLATFORM / INFERENCE (LAST)

> ⚠️ DO NOT BUILD THIS UNTIL PHASE 4 IS COMPLETE AND MODEL TRAINS SUCCESSFULLY

**Goal:** Wrap trained model into a simple web interface or script where a user can upload a CT image and receive segmentation output.

### 10.1 Inference Script

**File:** `src/inference.py`

```python
def predict(image_path, model_checkpoint_path):
    """
    1. Load image from path
    2. Preprocess: resize to 256x256, normalize, add batch dim
    3. Load model and checkpoint
    4. Forward pass with torch.no_grad()
    5. Apply sigmoid, threshold at 0.5
    6. Return: liver_mask, tumor_mask as numpy arrays
    """
```

### 10.2 Visualization Output

```python
def visualize_prediction(image_path, liver_mask, tumor_mask, save_path):
    """
    3-panel plot: [Original CT | Liver Mask | Tumor Mask]
    Save to save_path
    """
```

### 10.3 Optional Web UI (only if time permits)
- Use Gradio (simple 10-line UI)
- Input: upload image
- Output: side-by-side CT + predicted masks
- Do NOT build Flask/FastAPI — too complex for current stage

---

## 11. DEPENDENCIES

**`requirements.txt`**
```
torch>=2.0.0
torchvision>=0.15.0
pandas>=1.5.0
Pillow>=9.0.0
matplotlib>=3.5.0
numpy>=1.23.0
tqdm>=4.64.0
```

---

## 12. WHAT AI AGENT MUST NOT DO

```
❌ Do not re-implement the Dataset class or DataLoader (already done)
❌ Do not add argparse to any file — use CONFIG dict only
❌ Do not preprocess and save dataset to disk (low disk space)
❌ Do not use TensorFlow or Keras
❌ Do not add sigmoid inside model forward() — only in loss/inference
❌ Do not add augmentation yet (Phase 4 only — keep it simple)
❌ Do not build Phase 6 before Phase 4 works
❌ Do not download full dataset to local machine
❌ Do not use WidthType or any Word document libraries
❌ Do not use plt.show() in scripts — always save figures to results/
```

---

## 13. WHAT AI AGENT MUST ALWAYS DO

```
✅ Write comments explaining WHAT each block does and WHY
✅ Use simple English comments — this is a beginner learning project
✅ Print shapes at key points: print(f"Encoder block 1 output: {x.shape}")
✅ Test each file independently with __main__ block
✅ Handle device placement: .to(device) for model and data
✅ Use tqdm for progress bars in training loop
✅ Save all plots to results/ folder
✅ Separate each concern into its own file (model, loss, train, etc.)
✅ Follow the folder structure in Section 3 exactly
```

---

## 14. SUCCESS CRITERIA PER PHASE

| Phase | Success Condition |
|---|---|
| Phase 1 (EDA) | CSV stats printed, 5 sample images saved to results/eda/ |
| Phase 2 (Model) | `python src/models/unet.py` prints correct input/output shapes |
| Phase 3 (Loss) | Loss computes without error on random tensors |
| Phase 4 (Training) | Model trains for 5+ epochs, val dice > 0.3 for liver |
| Phase 5 (Cloud) | Notebook runs end-to-end on Kaggle without errors |
| Phase 6 (Platform) | Single image → segmentation mask in under 5 seconds |

---

## 15. GLOSSARY (FOR AI AGENT CONTEXT)

| Term | Meaning |
|---|---|
| U-Net | Encoder-decoder segmentation network with skip connections |
| Skip connection | Feature map copied from encoder and merged into decoder |
| Dice Loss | Loss based on overlap between predicted and true mask |
| BCE | Binary Cross Entropy — pixel-wise binary classification loss |
| Logits | Raw model output before sigmoid activation |
| Sigmoid | Function that squashes any number to 0–1 range |
| Encoder | Downsampling path — extracts features, reduces image size |
| Decoder | Upsampling path — recovers spatial size, produces mask |
| Bottleneck | Deepest layer in U-Net — smallest spatial size, most features |
| DataLoader | PyTorch utility that feeds batches of data to the model |
| Checkpoint | Saved model weights file (.pth) |
| Dice Score | Metric (0–1): how much predicted mask overlaps true mask |
| LiTS | Liver Tumor Segmentation — name of the dataset |
| CT Slice | Single 2D image from a 3D CT scan |
| Liver mask | Binary image: 1 where liver is, 0 elsewhere |
| Tumor mask | Binary image: 1 where tumor is, 0 elsewhere |

---

*End of PRD — liver_ai_project v1.0*
