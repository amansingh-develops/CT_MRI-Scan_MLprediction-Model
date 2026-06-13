# train_cnn.py
# This script trains a simple CNN (Convolutional Neural Network) on your CT images
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from src.preprocessing.preprocess_image import preprocess_ct_image

# ---------- Load images ----------

image_folder = "data/processed/images"
images = []

for file_name in os.listdir(image_folder):
    if file_name.endswith(".png"):
        file_path = os.path.join(image_folder, file_name)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = preprocess_ct_image(img)
        images.append(img)

images = np.array(images)
images = images[:, np.newaxis, :, :]  # (N, 1, 256, 256)

# ---------- Fake labels (learning stage) ----------

num_images = images.shape[0]
labels = np.zeros(num_images)
labels[num_images // 2:] = 1
labels = labels.astype(np.float32)

# ---------- Train / Validation split ----------

X_train, X_val, y_train, y_val = train_test_split(
    images, labels,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# ---------- Define CNN model ----------

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Conv Layer 1: input channels=1, output channels=8
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        # Conv Layer 2: input=8, output=16
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 64 * 64, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Second conv block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Fully connected
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

model = SimpleCNN()

# ---------- Loss and Optimizer ----------

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------- Training loop ----------

epochs = 15

for epoch in range(epochs):

    # -------- TRAINING --------
    model.train()   # training mode
    optimizer.zero_grad()

    train_outputs = model(X_train).squeeze()
    train_loss = criterion(train_outputs, y_train)

    train_loss.backward()
    optimizer.step()

    # -------- VALIDATION --------
    model.eval()    # evaluation mode
    with torch.no_grad():
        val_outputs = model(X_val).squeeze()
        val_loss = criterion(val_outputs, y_val)

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Loss: {train_loss.item():.4f} | "
        f"Val Loss: {val_loss.item():.4f}"
    )

print("Training with validation finished!")
