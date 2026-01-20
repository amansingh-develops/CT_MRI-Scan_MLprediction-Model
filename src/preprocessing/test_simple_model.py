# train_simple_model.py
# This script trains a very simple neural network on your images (learning purpose)

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------- Load images (same as before) ----------

image_folder = "data/processed/images"
images = []

for file_name in os.listdir(image_folder):
    if file_name.endswith(".png"):
        file_path = os.path.join(image_folder, file_name)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        images.append(img)

images = np.array(images)

# Add channel dimension (important for PyTorch)
# shape becomes: (N, 1, 256, 256)
images = images[:, np.newaxis, :, :]

# ---------- Create fake labels (only for learning) ----------

num_images = images.shape[0]

labels = np.zeros(num_images)

# First half = 0, second half = 1
labels[num_images // 2:] = 1

labels = labels.astype(np.float32)

# Convert to torch tensors
X = torch.tensor(images, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32)

# ---------- Define a very simple neural network ----------

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        # Flatten image: 256x256 -> 65536 numbers
        self.fc1 = nn.Linear(256 * 256, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = SimpleNet()

# ---------- Loss and Optimizer ----------

criterion = nn.BCELoss()        # Binary Cross Entropy Loss (for 0/1 classification)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------- Training loop ----------

epochs = 10

for epoch in range(epochs):
    optimizer.zero_grad()          # clear previous gradients
    outputs = model(X)             # model prediction
    outputs = outputs.squeeze()    # remove extra dimension

    loss = criterion(outputs, y)   # calculate loss
    loss.backward()                # backpropagation (compute gradients)
    optimizer.step()               # update weights

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("Training finished!")
