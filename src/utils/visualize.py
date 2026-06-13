import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_training_curves(log_csv_path, save_path="results/training_curves.png"):
    if not os.path.exists(log_csv_path):
        print("Log file not found.")
        return
        
    df = pd.read_csv(log_csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss')
    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss')
    ax1.set_title('Loss vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Dice plot
    ax2.plot(df['epoch'], df['train_dice_liver'], linestyle='--', color='blue', label='Train Liver Dice')
    ax2.plot(df['epoch'], df['val_dice_liver'], color='blue', label='Val Liver Dice')
    ax2.plot(df['epoch'], df['train_dice_tumor'], linestyle='--', color='red', label='Train Tumor Dice')
    ax2.plot(df['epoch'], df['val_dice_tumor'], color='red', label='Val Tumor Dice')
    ax2.set_title('Dice Score vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score (0-1)')
    ax2.legend()
    ax2.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")

def plot_prediction_sample(image, pred_mask, true_mask=None, save_path="results/prediction.png"):
    """
    image: [256, 256] numpy
    pred_mask: [2, 256, 256] numpy
    true_mask: [2, 256, 256] numpy (optional)
    """
    num_panels = 3 if true_mask is None else 5
    fig, axes = plt.subplots(1, num_panels, figsize=(num_panels*4, 4))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('CT Image')
    axes[0].axis('off')
    
    axes[1].imshow(pred_mask[0], cmap='gray')
    axes[1].set_title('Pred Liver')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask[1], cmap='gray')
    axes[2].set_title('Pred Tumor')
    axes[2].axis('off')
    
    if true_mask is not None:
        axes[3].imshow(true_mask[0], cmap='gray')
        axes[3].set_title('True Liver')
        axes[3].axis('off')
        
        axes[4].imshow(true_mask[1], cmap='gray')
        axes[4].set_title('True Tumor')
        axes[4].axis('off')
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
