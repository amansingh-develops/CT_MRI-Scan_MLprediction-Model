"""
Inference Pipeline for Liver & Tumor Segmentation
Based on: FasNet Paper (Nature Scientific Reports, 2025)

Usage:
    python -m src.inference --image "data/volume_0/slice_100.png"
    python -m src.inference --model "models/best_model.pth" --image "path/to/ct.png"
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from src.models.unet import UNet


class LiverTumorSegmenter:
    def __init__(self, model_path, device=None):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None else device
        )
        print(f"Using device: {self.device}")

        # Initialize Architecture
        self.model = UNet(in_ch=1, out_ch=2)

        # Load Weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {model_path} successfully.")

        # CLAHE object — same params as training DataLoader
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def predict_image(self, image_path, apply_clahe=True, threshold=0.5):
        """
        Predicts liver and tumor masks for a given CT slice image path.
        Returns: image(numpy 0-1), pred_liver(numpy binary), pred_tumor(numpy binary)
        """
        # 1. Load as grayscale uint8
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        # 2. Apply CLAHE (matching training pipeline exactly)
        if apply_clahe:
            image = self.clahe.apply(image)

        # 3. Resize to 256x256
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

        # 4. Normalize to 0-1
        image_float = image.astype(np.float32) / 255.0

        # 5. To Tensor
        img_tensor = torch.from_numpy(image_float).unsqueeze(0).unsqueeze(0).float()
        img_tensor = img_tensor.to(self.device)

        # 6. Predict — sigmoid applied here (NOT in model.forward)
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float().squeeze(0).cpu().numpy()  # [2, 256, 256]

        pred_liver = preds[0]
        pred_tumor = preds[1]

        return image_float, pred_liver, pred_tumor

    def visualize_prediction(self, image_path, apply_clahe=True, threshold=0.5, save_path=None):
        """
        Runs prediction and visualizes the results cleanly.
        """
        image, pred_liver, pred_tumor = self.predict_image(image_path, apply_clahe, threshold)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("CT Image (CLAHE)")
        axes[0].axis("off")

        axes[1].imshow(pred_liver, cmap="gray")
        axes[1].set_title("Predicted Liver")
        axes[1].axis("off")

        axes[2].imshow(pred_tumor, cmap="gray")
        axes[2].set_title("Predicted Tumor")
        axes[2].axis("off")

        # Overlay — Liver=Blue, Tumor=Red
        overlay = np.stack([image, image, image], axis=-1)  # [256,256,3]
        overlay = (overlay * 255).astype(np.uint8)
        overlay[pred_liver > 0.5, 2] = np.clip(overlay[pred_liver > 0.5, 2].astype(int) + 100, 0, 255).astype(np.uint8)
        overlay[pred_tumor > 0.5, 0] = 255
        overlay[pred_tumor > 0.5, 1] = 0
        overlay[pred_tumor > 0.5, 2] = 0

        axes[3].imshow(overlay)
        axes[3].set_title("Overlay (Blue=Liver, Red=Tumor)")
        axes[3].axis("off")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Liver AI Inference")
    parser.add_argument("--model", type=str, default="models/best_model.pth", help="Path to trained model")
    parser.add_argument("--image", type=str, required=True, help="Path to CT image slice")
    parser.add_argument("--save", type=str, default="results/inference.png", help="Path to save output")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE preprocessing")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Train the model first using: python -m src.training.train")
    elif not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
    else:
        segmenter = LiverTumorSegmenter(model_path=args.model)
        segmenter.visualize_prediction(
            args.image, apply_clahe=not args.no_clahe, save_path=args.save
        )
