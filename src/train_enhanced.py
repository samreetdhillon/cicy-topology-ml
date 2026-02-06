import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np


# Load your new enhanced data
X_enhanced = np.load('data/processed/X_enhanced.npy').astype(np.float32)
y = np.load('data/processed/y_hodge.npy').astype(np.float32)

# X_enhanced contains: [Flat Matrix (180 values) | Ambient Dim (1 value)]
X_img = X_enhanced[:, :180].reshape(-1, 1, 12, 15)
X_scalar = X_enhanced[:, 180:] # The last column

# Convert to Tensors
dataset = TensorDataset(
    torch.from_numpy(X_img), 
    torch.from_numpy(X_scalar), 
    torch.from_numpy(y).long()
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32)

# 3. Initialize Model, Loss, and Optimizer
from src.models.cnn_model import CICYClassifier
model = CICYClassifier()
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
print("Starting Training...")
for epoch in range(50):
    model.train()
    total_loss = 0
    for batch_img, batch_scalar, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_img, batch_scalar) # Pass both inputs
        # Assuming outputs is a tuple (h11_output, h21_output)
        loss = criterion(outputs[0], batch_y[:, 0].long()) + criterion(outputs[1], batch_y[:, 1].long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/50], Loss: {total_loss/len(train_loader):.4f}")

# 5. Save the Model
torch.save(model.state_dict(), 'models/cicy_cnn_v1.pt')
print("Model saved to models/cicy_cnn_v1.pt")