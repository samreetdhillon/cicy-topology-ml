"""
Enhanced Training Script for CICY Classification
Trains a CNN with scalar geometric features.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

from src.models.cnn_model import CICYClassifier

# Configuration
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
LAMBDA_H21 = 1.2
TRAIN_SPLIT = 0.8
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Load enhanced dataset
X_enhanced = np.load("data/processed/X_enhanced.npy").astype(np.float32)
y = np.load("data/processed/y_hodge.npy").astype(np.int64)

# Split enhanced input
X_img = X_enhanced[:, :180].reshape(-1, 1, 12, 15)
X_scalar = X_enhanced[:, 180:]  # scalar geometric features

# Convert to tensors
dataset = TensorDataset(
    torch.from_numpy(X_img),
    torch.from_numpy(X_scalar),
    torch.from_numpy(y)
)

# Train / Test split
train_size = int(TRAIN_SPLIT * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Model, loss, optimizer
model = CICYClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
print("Starting training...\n")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for batch_img, batch_scalar, batch_y in train_loader:
        optimizer.zero_grad()

        out_h11, out_h21 = model(batch_img, batch_scalar)

        loss_h11 = criterion(out_h11, batch_y[:, 0])
        loss_h21 = criterion(out_h21, batch_y[:, 1])

        # loss = loss_h11 + loss_h21
        loss = loss_h11 + (LAMBDA_H21 * loss_h21)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Loss: {avg_loss:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/cicy_cnn_v1.pt")

print("\nTraining complete.")
print("Model saved to models/cicy_cnn_v1.pt")
