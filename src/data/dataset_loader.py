import os
import numpy as np
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset


class LITSDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, apply_clahe=False):
        """
        csv_file : path to lits_train.csv
        root_dir : folder containing the actual PNG files (e.g. .../dataset_6)
        transform : optional preprocessing
        apply_clahe : apply CLAHE to CT images
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.apply_clahe = apply_clahe
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Just grab the filename — ignore all folder/prefix garbage from CSV
        img_name = os.path.basename(row["filepath"])
        liver_name = os.path.basename(row["liver_maskpath"])
        tumor_name = os.path.basename(row["tumor_maskpath"])

        # Build path: root_dir + filename. That's it.
        image = cv2.imread(os.path.join(self.root_dir, img_name), cv2.IMREAD_GRAYSCALE)
        liver = cv2.imread(os.path.join(self.root_dir, liver_name), cv2.IMREAD_GRAYSCALE)
        tumor = cv2.imread(os.path.join(self.root_dir, tumor_name), cv2.IMREAD_GRAYSCALE)

        # If file missing, use blank image instead of crashing
        if image is None:
            image = np.zeros((256, 256), dtype=np.uint8)
        if liver is None:
            liver = np.zeros((256, 256), dtype=np.uint8)
        if tumor is None:
            tumor = np.zeros((256, 256), dtype=np.uint8)

        # CLAHE
        if self.apply_clahe:
            image = self.clahe.apply(image)

        # Resize
        image = cv2.resize(image, (256, 256))
        liver = cv2.resize(liver, (256, 256))
        tumor = cv2.resize(tumor, (256, 256))

        # Normalize
        image = image / 255.0
        liver = liver / 255.0
        tumor = tumor / 255.0

        # Tensors
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        liver = torch.tensor(liver, dtype=torch.float32).unsqueeze(0)
        liver = (liver > 0.5).float()

        tumor = torch.tensor(tumor, dtype=torch.float32).unsqueeze(0)
        tumor = (tumor > 0.5).float()

        mask = torch.cat([liver, tumor], dim=0)
        return image, mask