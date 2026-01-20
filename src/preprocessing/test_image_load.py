# test_image_load.py
# This script loads one image and shows it on the screen

import cv2              # OpenCV (Open Source Computer Vision) library for images
import matplotlib.pyplot as plt   # matplotlib = library to draw graphs and images

# Path to one sample image (change this after you create one PNG image)
image_path = "data/processed/images/sample.png"

# Read image in grayscale mode (black and white)
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image loaded correctly
if img is None:
    print("Image not found! Check the path.")
else:
    print("Image loaded successfully!")
    print("Image shape:", img.shape)   # shape = (height, width)

    # Show the image
    plt.imshow(img, cmap='gray')
    plt.title("CT Slice")
    plt.axis('off')
    plt.show()
