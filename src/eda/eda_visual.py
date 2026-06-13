import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

# Ensure results directory exists
os.makedirs("results/eda", exist_ok=True)

def visualize_dataset_samples(csv_path="data/lits_train.csv", data_root="data", num_samples=5):
    """Pick random slices and save 3-panel plots + overlay to results/eda/"""
    print(f"Loading {csv_path} for visual samples...")
    if not os.path.exists(csv_path):
        print(f"Missing {csv_path}. Cannot generate visual samples.")
        return

    df = pd.read_csv(csv_path)
    
    # Filter only those that actually have the image file accessible (locally)
    valid_indices = []
    
    print("Checking for accessible local images...")
    # random sampling to be fast
    sample_size_to_check = min(100, len(df))
    random_indices = random.sample(range(len(df)), sample_size_to_check)
    
    for idx in random_indices:
        img_path = df.iloc[idx]['filepath'].replace("../input/lits-png/", "")
        if os.path.exists(os.path.join(data_root, img_path)):
            valid_indices.append(idx)
            
    if not valid_indices:
        print("No local image files found. Skipping visual sampling.")
        return
        
    print(f"Found local images! Processing {num_samples} random samples...")
    chosen_indices = random.sample(valid_indices, min(num_samples, len(valid_indices)))
    
    for i, idx in enumerate(chosen_indices):
        row = df.iloc[idx]
        
        # Paths
        img_path = os.path.join(data_root, row['filepath'].replace("../input/lits-png/", ""))
        liver_path = os.path.join(data_root, row['liver_maskpath'].replace("../input/lits-png/", ""))
        tumor_path = os.path.join(data_root, row['tumor_maskpath'].replace("../input/lits-png/", ""))
        
        # Load
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        liver = cv2.imread(liver_path, cv2.IMREAD_GRAYSCALE)
        tumor = cv2.imread(tumor_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or liver is None or tumor is None: continue
        
        # Apply CLAHE to mirror the training pipeline precisely
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

            
        # Plot 4-panel
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f"CT Slice (Study {row['study_number']})")
        axes[0].axis('off')
        
        axes[1].imshow(liver, cmap='gray')
        axes[1].set_title("Liver Mask")
        axes[1].axis('off')
        
        axes[2].imshow(tumor, cmap='gray')
        axes[2].set_title("Tumor Mask")
        axes[2].axis('off')
        
        # Overlay heatmap
        # Liver = Blue, Tumor = Red
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Add blue tint for liver
        liver_bool = liver > 128
        overlay[liver_bool, 2] = np.clip(overlay[liver_bool, 2] + 100, 0, 255) # Add blue
        
        # Add red tint for tumor
        tumor_bool = tumor > 128
        overlay[tumor_bool, 0] = 255 # Make red channel max
        overlay[tumor_bool, 1] = 0
        overlay[tumor_bool, 2] = 0
        
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay (Blue=Liver, Red=Tumor)")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"results/eda/sample_{i+1}_study{row['study_number']}.png")
        plt.close()
        
    print(f"Saved {min(num_samples, len(chosen_indices))} 4-panel plots to results/eda/")
    
def plot_pixel_intensity_histogram(csv_path="data/lits_train.csv", data_root="data", num_samples=10):
    """Plot histogram of pixel intensity values from random images"""
    print("Generating pixel intensity histogram...")
    if not os.path.exists(csv_path): return
        
    df = pd.read_csv(csv_path)
    
    pixels = []
    
    for _ in range(num_samples * 5): # try up to 50 times
        idx = random.randint(0, len(df)-1)
        img_path = os.path.join(data_root, df.iloc[idx]['filepath'].replace("../input/lits-png/", ""))
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                pixels.extend(img.flatten())
                if len(pixels) >= num_samples * 256 * 256: # Got enough
                    break
                    
    if not pixels:
        return
        
    plt.figure(figsize=(10, 6))
    plt.hist(pixels, bins=50, color='gray', alpha=0.7)
    plt.title(f"Pixel Intensity Distribution ({num_samples} random CT slices)")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("results/eda/pixel_intensity_histogram.png")
    plt.close()
    print("Saved results/eda/pixel_intensity_histogram.png")

def plot_slice_type_distribution(csv_path="data/lits_train.csv"):
    """Plot bar chart of slice types (liver only, tumor only, both, neither)"""
    print("Generating slice type distribution chart...")
    if not os.path.exists(csv_path): return
        
    df = pd.read_csv(csv_path)
    if 'liver_mask_empty' not in df.columns or 'tumor_mask_empty' not in df.columns:
        return
        
    liver_only = len(df[(df['liver_mask_empty'] == False) & (df['tumor_mask_empty'] == True)])
    tumor_only = len(df[(df['liver_mask_empty'] == True) & (df['tumor_mask_empty'] == False)])
    both = len(df[(df['liver_mask_empty'] == False) & (df['tumor_mask_empty'] == False)])
    neither = len(df[(df['liver_mask_empty'] == True) & (df['tumor_mask_empty'] == True)])
    
    labels = ['Liver Only', 'Tumor Only', 'Both', 'Background Only']
    counts = [liver_only, tumor_only, both, neither]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts, color=['#3498db', '#e74c3c', '#9b59b6', '#95a5a6'])
    
    # Add exact numbers on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(counts)*0.01), 
                 f"{yval}", ha='center', va='bottom', fontweight='bold')
                 
    plt.title("Slice Type Distribution (Train Dataset)")
    plt.ylabel("Number of Slices")
    plt.savefig("results/eda/slice_distribution.png")
    plt.close()
    print("Saved results/eda/slice_distribution.png")

if __name__ == "__main__":
    plot_slice_type_distribution()
    visualize_dataset_samples()
    plot_pixel_intensity_histogram()
