import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from src.models.cnn_model import CICYClassifier

# 1. Load Data
X_enhanced = np.load('data/processed/X_enhanced.npy').astype(np.float32)
y_actual = np.load('data/processed/y_hodge.npy').astype(np.float32)

model = CICYClassifier()
model.load_state_dict(torch.load('models/cicy_cnn_v1.pt'))
model.eval()

# 2. Identify Errors
with torch.no_grad():
    X_img = torch.from_numpy(X_enhanced[:, :180].reshape(-1, 1, 12, 15))
    X_scalar = torch.from_numpy(X_enhanced[:, 180:])
    
    _, out_h21 = model(X_img, X_scalar)
    pred_h21 = torch.argmax(out_h21, dim=1).numpy()

# Calculate absolute distance of the error
errors = np.abs(pred_h21 - y_actual[:, 1])
worst_indices = np.argsort(errors)[-5:] # The 5 biggest misses

print("--- ERROR ANALYSIS: WORST OFFENDERS (h2,1) ---")
for idx in worst_indices:
    print(f"\nManifold Index: {idx}")
    print(f"Actual h2,1: {y_actual[idx, 1]} | Predicted h2,1: {pred_h21[idx]}")
    # Reshape the flat matrix back to 12x15 to see the 'image'
    matrix = X_enhanced[idx, :180].reshape(12, 15)
    print("Configuration Matrix (Non-zero part):")
    # Only print rows/cols that aren't all zero
    print(matrix[~np.all(matrix == 0, axis=1)][:, ~np.all(matrix == 0, axis=0)])