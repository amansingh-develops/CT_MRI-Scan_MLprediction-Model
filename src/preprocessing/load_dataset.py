# load_dataset.py
# This script loads many images from a folder and stores them in a list

import os                  # os = operating system library (for folders and files)
import cv2                 # OpenCV for reading images
import numpy as np         # numpy = numerical library (arrays, matrices)

# Folder where your images are stored
image_folder = "data/processed/images"

images = []    # this will store all image arrays

# Loop through all files in the folder
for file_name in os.listdir(image_folder):
    if file_name.endswith(".png"):   # only take PNG images
        file_path = os.path.join(image_folder, file_name)

        # Read image in grayscale
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Resize image to fixed size (256 x 256)
        img = cv2.resize(img, (256, 256))

        # Normalize pixel values from 0–255 to 0–1
        img = img / 255.0

        images.append(img)

print("Total images loaded:", len(images))

# Convert list to numpy array (important for ML)
images = np.array(images)

print("Dataset shape:", images.shape)
