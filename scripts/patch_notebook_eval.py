"""
Patches livertumor-model.ipynb:
  - Replaces Cell 7 (basic predictions) with a 3-cell comprehensive evaluation suite
  - Cell 7: Full metrics on entire val set (Dice/IoU/Acc/Precision/Recall/F1)
  - Cell 8: Detailed side-by-side predictions with overlays + per-sample Dice
  - Cell 9: Publication-quality metrics dashboard
"""
import json

NB_PATH = "livertumor-model.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ── Find and remove old Cell 7 + footer ──────────────────────
# Cell 7 markdown starts with "### Cell 7"
# We remove everything from that markdown cell onward
cut_idx = None
for i, cell in enumerate(cells):
    src = "".join(cell.get("source", []))
    if "Cell 7" in src and cell["cell_type"] == "markdown":
        cut_idx = i
        break

if cut_idx is None:
    print("ERROR: Could not find Cell 7 markdown!")
    exit(1)

print(f"Found Cell 7 at index {cut_idx}")
print(f"Removing {len(cells) - cut_idx} cells from index {cut_idx} onward")

# Keep only cells before Cell 7
cells = cells[:cut_idx]

# ── NEW CELLS ─────────────────────────────────────────────────

# --- Cell 7 Markdown ---
cell7_md = {
    "cell_type": "markdown",
    "source": "### Cell 7 — Full Model Evaluation (Metrics on Entire Val Set)\n\nComputes **every metric** on the full validation set:\n- Dice Score (per channel + mean)\n- IoU / Jaccard Index (per channel + mean)\n- Pixel Accuracy (global + per channel)\n- Precision, Recall, F1 (per channel)\n- Confusion matrix counts (TP, FP, FN, TN)",
    "metadata": {}
}

# --- Cell 7 Code ---
cell7_code = {
    "cell_type": "code",
    "source": r'''print("=" * 70)
print("  📊 FULL MODEL EVALUATION — Best Checkpoint on Val Set")
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
# Channel 0 = Liver, Channel 1 = Tumor
tp = torch.zeros(2, dtype=torch.float64)
fp = torch.zeros(2, dtype=torch.float64)
fn = torch.zeros(2, dtype=torch.float64)
tn = torch.zeros(2, dtype=torch.float64)

# Dice accumulators (for global Dice, not batch-averaged)
inter_sum = torch.zeros(2, dtype=torch.float64)
pred_sum  = torch.zeros(2, dtype=torch.float64)
true_sum  = torch.zeros(2, dtype=torch.float64)

# Per-sample dice lists
sample_dice_liver = []
sample_dice_tumor = []

total_pixels = 0
correct_pixels = 0  # for pixel accuracy
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
            p = preds[:, ch].reshape(B, -1)  # [B, H*W]
            t = targets[:, ch].reshape(B, -1)

            tp[ch] += (p * t).sum().item()
            fp[ch] += (p * (1 - t)).sum().item()
            fn[ch] += ((1 - p) * t).sum().item()
            tn[ch] += ((1 - p) * (1 - t)).sum().item()

            inter_sum[ch] += (p * t).sum().item()
            pred_sum[ch]  += p.sum().item()
            true_sum[ch]  += t.sum().item()

        # Per-sample Dice
        for b in range(B):
            for ch, lst in [(0, sample_dice_liver), (1, sample_dice_tumor)]:
                p = preds[b, ch].reshape(-1)
                t = targets[b, ch].reshape(-1)
                inter = (p * t).sum().item()
                denom = p.sum().item() + t.sum().item()
                if denom > 0:
                    lst.append((2 * inter + smooth) / (denom + smooth))
                else:
                    lst.append(1.0)  # both empty = perfect

        # Pixel accuracy (across both channels)
        total_pixels += preds.numel()
        correct_pixels += (preds == targets).sum().item()

print("\n" + "=" * 70)
print("  RESULTS — Full Validation Set")
print("=" * 70)

# ── Compute metrics ───────────────────────────────────────────
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

# Mean metrics
mean_dice = (metrics['Liver']['Dice'] + metrics['Tumor']['Dice']) / 2
mean_iou  = (metrics['Liver']['IoU'] + metrics['Tumor']['IoU']) / 2
pixel_acc = correct_pixels / total_pixels

print(f"\n  ── Overall ──")
print(f"  Mean Dice:       {mean_dice:.4f}")
print(f"  Mean IoU:        {mean_iou:.4f}")
print(f"  Pixel Accuracy:  {pixel_acc:.4f} ({pixel_acc*100:.2f}%)")

# Per-sample Dice stats
print(f"\n  ── Per-Sample Dice Distribution ──")
for name, lst in [('Liver', sample_dice_liver), ('Tumor', sample_dice_tumor)]:
    arr = np.array(lst)
    non_trivial = arr[arr < 1.0]  # exclude empty-matches-empty
    print(f"  {name}: mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
          f"std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")
    if len(non_trivial) > 0:
        print(f"    Non-trivial (has GT): mean={non_trivial.mean():.4f}, "
              f"median={np.median(non_trivial):.4f}, n={len(non_trivial)}")

# Save metrics to CSV
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
print("=" * 70)''',
    "metadata": {"trusted": True},
    "outputs": [],
    "execution_count": None
}

# --- Cell 8 Markdown ---
cell8_md = {
    "cell_type": "markdown",
    "source": "### Cell 8 — Detailed Side-by-Side Predictions (Real vs Predicted)\n\nShows 8 carefully selected samples with:\n- CT scan | Ground Truth | Prediction | Color Overlay\n- Per-sample Dice scores annotated on each image\n- Green = Liver, Red = Tumor in overlays\n- TP/FP/FN color-coded comparison",
    "metadata": {}
}

# --- Cell 8 Code ---
cell8_code = {
    "cell_type": "code",
    "source": r'''print("=" * 70)
print("  🖼️  DETAILED PREDICTION VISUALIZATIONS")
print("=" * 70)

# ── Helper: Create color overlay ──────────────────────────────
def make_overlay(ct_slice, true_liver, true_tumor, pred_liver, pred_tumor):
    """Create a color overlay showing TP (green), FP (yellow), FN (red)."""
    h, w = ct_slice.shape
    ct_rgb = np.stack([ct_slice * 255] * 3, axis=-1).astype(np.uint8)
    overlay = ct_rgb.copy().astype(np.float32)

    # Liver: TP=green, FP=yellow, FN=red
    liver_tp = (true_liver > 0) & (pred_liver > 0)
    liver_fp = (true_liver == 0) & (pred_liver > 0)
    liver_fn = (true_liver > 0) & (pred_liver == 0)

    overlay[liver_tp] = overlay[liver_tp] * 0.4 + np.array([0, 200, 0]) * 0.6
    overlay[liver_fp] = overlay[liver_fp] * 0.4 + np.array([255, 200, 0]) * 0.6
    overlay[liver_fn] = overlay[liver_fn] * 0.4 + np.array([200, 0, 0]) * 0.6

    # Tumor: TP=cyan, FP=magenta, FN=orange
    tumor_tp = (true_tumor > 0) & (pred_tumor > 0)
    tumor_fp = (true_tumor == 0) & (pred_tumor > 0)
    tumor_fn = (true_tumor > 0) & (pred_tumor == 0)

    overlay[tumor_tp] = overlay[tumor_tp] * 0.3 + np.array([0, 255, 255]) * 0.7
    overlay[tumor_fp] = overlay[tumor_fp] * 0.3 + np.array([255, 0, 255]) * 0.7
    overlay[tumor_fn] = overlay[tumor_fn] * 0.3 + np.array([255, 140, 0]) * 0.7

    return np.clip(overlay, 0, 255).astype(np.uint8)

# ── Select 8 diverse samples ─────────────────────────────────
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

# ── Generate predictions and visualize ────────────────────────
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

    # Per-sample dice
    def sample_dice(p, t):
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        if denom == 0: return 1.0
        return (2 * inter + 1e-6) / (denom + 1e-6)

    dl = sample_dice(pl, tl)
    dt = sample_dice(pt, tt)

    # Create overlays
    overlay_gt = make_overlay(ct, tl, tt, tl, tt)
    overlay_pred = make_overlay(ct, tl, tt, pl, pt)

    # Plot 8 columns
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

    # Difference map
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

# ── Histogram of per-sample Dice ──────────────────────────────
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
print("💾 Saved to results/dice_distribution.png")''',
    "metadata": {"trusted": True},
    "outputs": [],
    "execution_count": None
}

# --- Cell 9 Markdown ---
cell9_md = {
    "cell_type": "markdown",
    "source": "### Cell 9 — Metrics Dashboard (Publication-Quality Summary)\n\nA single image summarizing all metrics, training info, and model performance — ready for reports or presentations.",
    "metadata": {}
}

# --- Cell 9 Code ---
cell9_code = {
    "cell_type": "code",
    "source": r'''print("=" * 70)
print("  📋 GENERATING METRICS DASHBOARD")
print("=" * 70)

fig = plt.figure(figsize=(24, 16))
fig.patch.set_facecolor('#1a1a2e')

# ── Title ─────────────────────────────────────────────────────
fig.suptitle('Liver & Tumor Segmentation — Model Evaluation Dashboard',
             fontsize=22, fontweight='bold', color='white', y=0.98)

# ── ROW 1: Metric Cards ──────────────────────────────────────
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

# ── ROW 2 LEFT: Training Curves ──────────────────────────────
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

# ── ROW 2 RIGHT: Per-Channel Detail Table ─────────────────────
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

# ── ROW 3 LEFT: Confusion Matrix Bars ─────────────────────────
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

# ── ROW 3 RIGHT: Model Info ───────────────────────────────────
ax5 = fig.add_axes([0.55, 0.05, 0.42, 0.28])
ax5.set_facecolor('#16213e')
ax5.set_xticks([]); ax5.set_yticks([])
ax5.set_title('Model Configuration', fontsize=13, color='white', pad=10)
for spine in ax5.spines.values(): spine.set_color('#34495e')

info_lines = [
    f"Architecture:  Standard 2D U-Net (64>128>256>512>1024)",
    f"Parameters:    {sum(p.numel() for p in model.parameters()):,}",
    f"Loss Function: 0.5 x BCE + 0.5 x Dice (no pos_weight)",
    f"Optimizer:     Adam (lr=1e-4)",
    f"Batch Size:    32",
    f"Best Epoch:    {ckpt['epoch']}",
    f"Val Loss:      {ckpt['val_loss']:.6f}",
    f"Dataset:       LiTS17 ({len(eval_ds):,} val samples)",
    f"Input Size:    256 x 256 (grayscale)",
    f"AMP:           Mixed Precision (fp16/fp32)",
]
for i, line in enumerate(info_lines):
    ax5.text(0.05, 0.92 - i * 0.095, line, ha='left', va='center',
             fontsize=10, color='#bdc3c7', transform=ax5.transAxes,
             fontfamily='monospace')

plt.savefig('results/evaluation_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
plt.show()
print("\n💾 Saved to results/evaluation_dashboard.png")

# ── Final Summary ─────────────────────────────────────────────
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
print("=" * 70)''',
    "metadata": {"trusted": True},
    "outputs": [],
    "execution_count": None
}

# --- Footer Markdown ---
footer_md = {
    "cell_type": "markdown",
    "source": "---\n\n### Target Metrics (from FasNet paper)\n\n| Stage | Dice Liver | Dice Tumor | Notes |\n|---|---|---|---|\n| First run (base U-Net) | > 0.60 | > 0.30 | Model is learning |\n| Good base U-Net | ~0.74 | ~0.50 | Comparable to baselines |\n| After adding attention | ~0.85 | ~0.70 | Paper-level |\n| FasNet (ResNet50+VGG16) | 0.8766 | — | State of the art |\n\n### Evaluation Outputs\n\n| File | Description |\n|---|---|\n| `evaluation_metrics.csv` | Dice, IoU, Precision, Recall, F1, Accuracy per channel |\n| `detailed_predictions.png` | 8 samples: CT → True → Pred → Overlay → Diff map |\n| `dice_distribution.png` | Per-sample Dice histograms for Liver and Tumor |\n| `evaluation_dashboard.png` | Publication-quality summary with all metrics |\n| `training_curves.png` | Loss and Dice curves over training |\n\n**Next steps after base U-Net:**\n1. Add Attention Gates → expected +5-10% Dice\n2. Pretrained ResNet34 encoder → expected +5% Dice",
    "metadata": {}
}

# ── Assemble ──────────────────────────────────────────────────
new_cells = [cell7_md, cell7_code, cell8_md, cell8_code, cell9_md, cell9_code, footer_md]
cells.extend(new_cells)
nb["cells"] = cells

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅ Notebook patched successfully!")
print(f"   Removed old Cell 7 + footer")
print(f"   Added {len(new_cells)} new cells:")
print(f"     - Cell 7: Full Metrics Evaluation (Dice/IoU/Acc/Prec/Recall/F1)")
print(f"     - Cell 8: Detailed Predictions (8 samples, overlays, diff maps)")
print(f"     - Cell 9: Metrics Dashboard (publication-quality)")
print(f"     - Footer: Target metrics + output file descriptions")
