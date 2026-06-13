# preprocess_image.py
# This file contains preprocessing functions for CT images

import cv2
import numpy as np

def preprocess_ct_image(img):
    """
    Preprocess a single CT image.
    Input:
        img: grayscale image (numpy array)
    Output:
        processed image
    """

    # 1) Resize image to fixed size
    img = cv2.resize(img, (256, 256))

    # 2) Normalize pixel values to 0–1
    img = img / 255.0

    # 3) Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Convert to uint8 because CLAHE expects 0–255
    img_uint8 = (img * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,     # controls contrast strength
        tileGridSize=(8, 8)  # local region size
    )

    img_clahe = clahe.apply(img_uint8)

    # 4) Convert back to 0–1 range
    img_clahe = img_clahe / 255.0

    return img_clahe
